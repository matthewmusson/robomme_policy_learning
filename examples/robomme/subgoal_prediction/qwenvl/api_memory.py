"""
Memory-augmented Qwen3-VL subgoal predictor for the canonical RoboMME eval.

Parallel to `api.py` (SimpleSG, GroundSG) and `api_memer.py` (MemER).
Supports two prompt variants matching the v2 training data built by
`project/build_memory_subgoal_jsonl.py`:

  * with_history=True  -> "memory + history" (history line preserved)
  * with_history=False -> "memory only" (history line dropped)

Both:
  - Use a single front-view image per inference (matches SimpleSG).
  - Inject a "Current memory: <mem>{m_t}</mem>" line into the user prompt.
  - Expect the assistant to output `{subgoal}\\n<mem>{m_{t+1}}</mem>`.
  - Maintain `self.current_memory` across turns; parse `<mem>...</mem>`
    out of each response to update it and to feed the VLA a clean
    subgoal string (no <mem> tags).

Drop-in for the QwenVLSubgoalPredictor wrapper in `subgoal_predictor.py`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import List

import imageio
import numpy as np
import pprint

os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '256')
os.environ.setdefault('VIDEO_MAX_TOKEN_NUM', '64')
os.environ.setdefault('FPS_MAX_FRAMES', '10')

from swift.llm import PtEngine, InferRequest, RequestConfig


# Must match the build-time system prompt in build_memory_subgoal_jsonl.py
# for the corresponding memory_style. simple uses SYSTEM_PROMPT_SIMPLE;
# layout and grounded BOTH use SYSTEM_PROMPT_GROUNDED at training/runtime
# (they predict grounded subgoals — the memory text format is what differs).
# dual uses SYSTEM_PROMPT_DUAL (action memory + spatial memory in two tags).
SYSTEM_PROMPT_SIMPLE = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of language subgoals. You also maintain a compact "
    "memory summary of relevant past events between turns."
)
SYSTEM_PROMPT_GROUNDED = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of grounded language subgoals. You also maintain a "
    "compact memory summary of relevant past events between turns."
)
SYSTEM_PROMPT_DUAL = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of language subgoals. You also maintain TWO compact "
    "memories between turns: a textual action memory of what has happened so far "
    "(wrapped in <mem>...</mem> tags) and a spatial memory describing the current "
    "scene layout (wrapped in <spatial_mem>...</spatial_mem> tags). "
    "Output the next subgoal, then both memories, in that order."
)
SYSTEM_PROMPT_DUAL_GROUNDED = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of grounded language subgoals. You also maintain TWO compact "
    "memories between turns: a textual action memory of what has happened so far "
    "(wrapped in <mem>...</mem> tags) and a spatial memory describing the current "
    "scene layout (wrapped in <spatial_mem>...</spatial_mem> tags). "
    "Output the next subgoal, then both memories, in that order."
)
# Back-compat alias; some sites still import SYSTEM_PROMPT directly.
SYSTEM_PROMPT = SYSTEM_PROMPT_SIMPLE

MEMORY_STYLES = ("simple", "layout", "grounded", "dual")
DUAL_TEXT_STYLES = ("simple", "present", "grounded", "layout")


def _style_to_runtime_config(memory_style: str, text_style: str = "present") -> dict:
    """At inference time, what system prompt + wording does this memory_style
    use. Mirrors the build-time config in build_memory_subgoal_jsonl.py.

    layout vs grounded share the same wording at runtime — only the LoRA
    adapter weights know how to format the memory output differently.
    dual uses its own system prompt and an extra <spatial_mem> block.

    `text_style` is only meaningful for memory_style="dual"; it selects
    which textual action memory style the dual adapter was trained with
    and therefore which subgoal target it produces. Must match the value
    passed to build_memory_subgoal_jsonl.py at training time."""
    if memory_style == "simple":
        return {
            "system_prompt": SYSTEM_PROMPT_SIMPLE,
            "sg_word":       "language",
            "subgoal_type":  "simple_subgoal",
            "uses_spatial":  False,
        }
    if memory_style in ("layout", "grounded"):
        return {
            "system_prompt": SYSTEM_PROMPT_GROUNDED,
            "sg_word":       "grounded language",
            "subgoal_type":  "grounded_subgoal",
            "uses_spatial":  False,
        }
    if memory_style == "dual":
        if text_style not in DUAL_TEXT_STYLES:
            raise ValueError(
                f"text_style must be one of {DUAL_TEXT_STYLES}, got {text_style!r}"
            )
        if text_style in ("grounded", "layout"):
            return {
                "system_prompt": SYSTEM_PROMPT_DUAL_GROUNDED,
                "sg_word":       "grounded language",
                "subgoal_type":  "grounded_subgoal",
                "uses_spatial":  True,
            }
        return {
            "system_prompt": SYSTEM_PROMPT_DUAL,
            "sg_word":       "language",
            "subgoal_type":  "simple_subgoal",
            "uses_spatial":  True,
        }
    raise ValueError(
        f"memory_style must be one of {MEMORY_STYLES}, got {memory_style!r}"
    )

_MEM_BLOCK_RE = re.compile(r"<mem>(.*?)</mem>", re.DOTALL)
_SPATIAL_MEM_BLOCK_RE = re.compile(r"<spatial_mem>(.*?)</spatial_mem>", re.DOTALL)
# Open-tag-only fallbacks. The grounded/layout dual adapters often drop the
# closing tag (because grounded coords like `<63, 136>` inside the mem text
# confuse the model's stop-token distribution). When the strict close-paired
# regex misses, we still want to extract the partial content rather than
# treat the entire response as the subgoal — that downstream-cascades into
# a malformed subgoal that the VLA tries to act on AND a video-encode crash
# because the rendered text overlay grows past frame height. The fallback
# extracts everything between <mem> and the next plausible boundary
# (</mem> | <spatial_mem> | end-of-string).
_MEM_OPEN_RE = re.compile(r"<mem>")
_SPATIAL_OPEN_RE = re.compile(r"<spatial_mem>")
_MEM_CLOSE_RE = re.compile(r"</mem>")
_SPATIAL_CLOSE_RE = re.compile(r"</spatial_mem>")
_INITIAL_MEMORY = "No actions had been completed yet."
_INITIAL_SPATIAL_MEMORY = "No observations yet."


class Qwen3VLModelMemory:
    """Memory-augmented subgoal predictor.

    Args:
        adapter_path: path to LoRA adapter (e.g.
            `/vol/memory_vlm_v2/from_simplesg/v*/checkpoint-*/`).
        with_history: if True, include the SimpleSG-style numbered
            history line; if False, drop it (memory-only variant).
        memory_style: one of MEMORY_STYLES = (simple, layout, grounded, dual).
            Picks the system prompt + user-prompt wording that this adapter
            was trained against. Must match the value used by
            build_memory_subgoal_jsonl.py when the training JSONL was made.
        text_style: only meaningful for memory_style="dual". One of
            DUAL_TEXT_STYLES = (simple, present, grounded, layout). Picks
            the textual action memory variant the adapter was trained
            with. Must match build_memory_subgoal_jsonl.py's --text-style.
    """

    def __init__(
        self,
        adapter_path: str,
        with_history: bool = True,
        memory_style: str = "simple",
        text_style: str = "present",
    ):
        cfg = _style_to_runtime_config(memory_style, text_style)
        self.memory_style = memory_style
        self.text_style = text_style
        style_tag = (
            f"{memory_style}_{text_style}" if memory_style == "dual" and text_style != "present"
            else memory_style
        )
        self.model_name = (
            f"qwenvl_memory_{style_tag}_history" if with_history
            else f"qwenvl_memory_{style_tag}_only"
        )
        self.subgoal_type = cfg["subgoal_type"]
        self._sg_word = cfg["sg_word"]
        self.uses_spatial = cfg["uses_spatial"]
        self.with_history = with_history
        self.image_size = (256, 256)
        self.system_prompt = cfg["system_prompt"]

        print(
            f"Loading Qwen3-VL-4B-Instruct adapter (memory_style={memory_style}, "
            f"with_history={with_history}) from {adapter_path}"
        )
        # Prefer the pre-downloaded base on /vol/qwen3-vl-4b-instruct (saves
        # a ~9 GB ModelScope fetch per cold start). Fall back to the hub id
        # if the path isn't there (e.g. running off-Modal).
        base_model = (
            "/vol/qwen3-vl-4b-instruct"
            if os.path.isdir("/vol/qwen3-vl-4b-instruct")
            else "Qwen/Qwen3-VL-4B-Instruct"
        )
        print(f"[Qwen3VLModelMemory] base model: {base_model}")
        self.engine = PtEngine(
            model_id_or_path=base_model,
            adapters=[adapter_path],
            attn_impl='sdpa',   # matches the v2 training recipe (sdpa, not flash-attn)
        )

    # -------------------------------------------------------------------------
    # Episode lifecycle (mirrors api.py:Qwen3VLModel)
    # -------------------------------------------------------------------------
    def start_new_episode(
        self,
        save_dir: str,
        video_query: List[np.ndarray] | None,
        task_goal: str = None,
    ) -> None:
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        ep_name = os.path.basename(save_dir)
        self.save_json_path = os.path.join(
            os.path.dirname(save_dir), f"{ep_name}_QwenVLMemory_log.jsonl"
        )

        if video_query is not None and len(video_query) > 0:
            imageio.mimsave(os.path.join(self.save_dir, "step_0_video.mp4"),
                            video_query, fps=30)
            self.video_path = os.path.join(self.save_dir, "step_0_video.mp4")
        else:
            self.video_path = None

        self.task_goal = task_goal
        self.history_simple_subgoals: list[str] = []
        self.current_memory: str = _INITIAL_MEMORY
        self.current_spatial_memory: str = (
            _INITIAL_SPATIAL_MEMORY if self.uses_spatial else ""
        )
        self.last_response: str | None = None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _wrap_history_subgoals(self, subgoals) -> str:
        return "; ".join(f"{i+1}. {s}" for i, s in enumerate(subgoals))

    def _parse_memory_response(self, raw: str) -> tuple[str, str]:
        """Split a `{subgoal}\\n<mem>{memory}</mem>` response.

        Returns (clean_subgoal_text, updated_action_memory). The <spatial_mem>
        block (if present, only for dual style) is consumed separately via
        _parse_spatial_memory.

        Lenient strategy (matters a lot for grounded/layout dual adapters):
          1. Strict match: `<mem>...</mem>` → use those bounds.
          2. Fallback if `<mem>` is present but `</mem>` is missing:
             treat content from `<mem>` to the next plausible boundary
             (`<spatial_mem>` or end-of-string) as the action memory.
             Subgoal is still everything BEFORE `<mem>`.
          3. No `<mem>` open tag at all: keep prior memory, use whole text
             as subgoal (legacy behavior — last-resort fallback).

        The fallback (2) is critical because the grounded/layout dual adapters
        regularly emit `<mem>...<spatial_mem>...</spatial_mem>` (forgetting
        the `</mem>` close). Under the strict regex this fell through to (3),
        which made the subgoal a 200+ char malformed string. That long string
        then crashed `RolloutRecorder.record()` downstream because the text
        overlay area grew past the video frame height, producing the
        "All images in a movie should have same size" errors we saw.
        """
        match = _MEM_BLOCK_RE.search(raw)
        if match:
            subgoal_text = raw[: match.start()].strip()
            memory_text = match.group(1).strip()
            return subgoal_text, memory_text

        open_m = _MEM_OPEN_RE.search(raw)
        if open_m:
            mem_start = open_m.end()
            spatial_open = _SPATIAL_OPEN_RE.search(raw, mem_start)
            mem_end = spatial_open.start() if spatial_open else len(raw)
            subgoal_text = raw[: open_m.start()].strip()
            memory_text = raw[mem_start:mem_end].strip()
            print(
                f"[Qwen3VLModelMemory] WARNING: missing </mem> close tag; "
                f"recovered subgoal={subgoal_text!r} mem_len={len(memory_text)}"
            )
            return subgoal_text, memory_text

        print(f"[Qwen3VLModelMemory] WARNING: no <mem> in response; using whole text as subgoal: {raw!r}")
        return raw.strip(), self.current_memory

    def _parse_spatial_memory(self, raw: str) -> str:
        """Extract the <spatial_mem>...</spatial_mem> block, or keep prior.

        Lenient: if `<spatial_mem>` is present but `</spatial_mem>` is missing,
        accept content from `<spatial_mem>` to end-of-string.
        """
        m = _SPATIAL_MEM_BLOCK_RE.search(raw)
        if m:
            return m.group(1).strip()

        open_m = _SPATIAL_OPEN_RE.search(raw)
        if open_m:
            spatial_text = raw[open_m.end():].strip()
            if self.uses_spatial:
                print(
                    f"[Qwen3VLModelMemory] WARNING: missing </spatial_mem> "
                    f"close tag; recovered spatial_len={len(spatial_text)}"
                )
            return spatial_text

        if self.uses_spatial:
            print(
                f"[Qwen3VLModelMemory] WARNING: no <spatial_mem> in dual "
                f"response; keeping prior spatial: {raw!r}"
            )
        return self.current_spatial_memory

    def update_history_subgoals(self, raw_response: str) -> None:
        """Update history + current_memory (+ spatial for dual) from a raw model response."""
        subgoal_text, memory_text = self._parse_memory_response(raw_response)
        # Dedupe-while-preserving-order, matching api.py's logic.
        if self.history_simple_subgoals:
            if self.history_simple_subgoals[-1] != subgoal_text:
                self.history_simple_subgoals.append(subgoal_text)
        else:
            self.history_simple_subgoals.append(subgoal_text)
        self.current_memory = memory_text
        if self.uses_spatial:
            self.current_spatial_memory = self._parse_spatial_memory(raw_response)

    def _parse_subgoal_for_vla(self, raw_response: str) -> str:
        """Return the clean subgoal text (no <mem>/<spatial_mem> tags) to feed the VLA."""
        subgoal_text, _ = self._parse_memory_response(raw_response)
        return subgoal_text

    # -------------------------------------------------------------------------
    # Prompt construction
    # -------------------------------------------------------------------------
    def _user_prompt(self, video_prefix: str) -> str:
        parts = [f"{video_prefix}The task goal is: {self.task_goal}"]
        if self.with_history:
            if not self.history_simple_subgoals:
                parts.append("This is the initial turn for prediction")
            else:
                parts.append(
                    f"The history of previous predicted {self._sg_word} subgoals are: "
                    + self._wrap_history_subgoals(self.history_simple_subgoals)
                )
        if self.uses_spatial:
            parts.append(
                f"The current action memory (a brief summary of relevant past events): "
                f"<mem>{self.current_memory}</mem>"
            )
            parts.append(
                f"The current spatial memory (a description of the current scene): "
                f"<spatial_mem>{self.current_spatial_memory}</spatial_mem>"
            )
        else:
            parts.append(
                f"The current memory (a brief summary of relevant past events): "
                f"<mem>{self.current_memory}</mem>"
            )
        parts.append(
            f"<image>What's the next {self._sg_word} subgoal based on current observation?"
        )
        return "\n".join(parts)

    def prepare_infer_request(
        self, image_query: np.ndarray, step_idx: int
    ) -> InferRequest:
        image_path = os.path.join(self.save_dir, f"step_{step_idx}_image.png")
        imageio.imwrite(image_path, image_query)
        video_prefix = "<video>" if self.video_path else ""

        user_prompt = self._user_prompt(video_prefix)

        infer_request_dict = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "images": [image_path],
        }
        if self.video_path is not None:
            infer_request_dict["videos"] = [self.video_path]

        print("\n\n")
        pprint.pprint(infer_request_dict)
        with open(self.save_json_path, "a") as f:
            json.dump(infer_request_dict, f)
            f.write("\n")

        return InferRequest(**infer_request_dict)

    # -------------------------------------------------------------------------
    # Inference entry (mirrors api.py:Qwen3VLModel.call signature)
    # -------------------------------------------------------------------------
    def call(
        self,
        image_query: np.ndarray,
        step_idx: int,
        keep_period: int = 0,
    ) -> str:
        if step_idx <= keep_period and self.last_response is not None:
            response = self.last_response
        else:
            infer_request = self.prepare_infer_request(image_query, step_idx)
            # Token budget. simple/layout/grounded emit only one memory block
            # so 128 is plenty. dual emits BOTH <mem> and <spatial_mem> so it
            # needs more headroom — truncation in the middle of the closing
            # </spatial_mem> tag silently drops the new spatial memory.
            max_tokens = 384 if self.uses_spatial else 128
            response = self.engine.infer(
                [infer_request],
                request_config=RequestConfig(max_tokens=max_tokens, temperature=0),
            )
            response = response[0].choices[0].message.content

        print("Response (raw):", response)
        self.last_response = response
        self.update_history_subgoals(response)
        return self._parse_subgoal_for_vla(response)
