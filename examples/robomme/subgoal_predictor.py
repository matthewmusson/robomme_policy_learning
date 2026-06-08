from typing import Optional, Tuple, Any
from pathlib import Path

import os
import shutil
from env_runner import EnvRunner
from utils import EpisodeState, SUBGOAL_TYPES, TASK_WITH_VIDEO_DEMO

from subgoal_prediction.gemini.api import GeminiModel
from subgoal_prediction.gemini.prompts import (
    DEMO_TEXT_QUERY,
    IMAGE_TEXT_QUERY,
    VIDEO_TEXT_QUERY,
)

from subgoal_prediction.qwenvl.api import Qwen3VLModel
from subgoal_prediction.qwenvl.api_memer import Qwen3VLModelMemER
from subgoal_prediction.qwenvl.api_memory import Qwen3VLModelMemory


LONG_FIRST_ACTION_TASKS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg"
] # For Gemini only. Due to we found Gemini is very inconsistent for incremental video understanding, hard code to make it work better



class SubgoalPredictorBase:
    def __init__(
        self,
        args,
        save_dir: Path,
    ):
        self.args = args
        self.save_dir = save_dir
        self.video_buffer = []
        self.episode_dir: Optional[str] = None
        
        self.setup_api()

    def setup_api(self) -> None:
        pass

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        self.env_name = env_runner.env_id
        self.episode_id = env_runner.episode_id
        self.task_goal = env_runner.task_goal
        self.env_runner = env_runner

    def step(self, epstate: EpisodeState) -> None:
        pass

    def maybe_extend_video(self, images: list) -> None:
        pass

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # return (subgoal_str, has_api_error)
        raise NotImplementedError

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        pass

    def get_last_vlm_call(self) -> Optional[dict]:
        """Metadata about the most recent VLM forward, for structured
        tracing. Returns None for predictors that don't call a VLM
        (oracle / null), or until the first call has happened."""
        return None

    def get_prompt_template(self) -> Optional[dict]:
        """The system prompt + user-prompt template (with placeholders)
        used at inference time. Stamped once per episode at the top of
        trace.json so downstream analysis knows exactly what the VLM was
        being asked. Returns None for predictors that don't prompt a VLM."""
        return None


class NullSubgoalPredictor(SubgoalPredictorBase):
    def get_subgoal(self, *args, **kwargs) -> Tuple[Optional[str], bool]:
        return None, False
    

class GeminiSubgoalPredictor(SubgoalPredictorBase):
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.api = GeminiModel(
            save_dir=os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}"),
            task_id=self.env_name,
            model_name=self.args.gemini_model_name,
            task_goal=self.task_goal,
            subgoal_type=self.args.subgoal_type,
        )
        self.video_buffer.extend(epstate.image_buffer[:-1])
        print(f"[robomme] Gemini agent for {self.args.subgoal_type}, task {self.env_name}, episode {self.episode_id}, setup finished")

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])
    
    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if not self._should_call(count):
            return current_subgoal, False

        text_query = self._get_text_query(count)
        input_data = self.api.prepare_input_data(self.video_buffer, text_query, count)
        response, _ = self.api.call(input_data)
        self.video_buffer.clear()

        if response is None:
            return None, True

        subgoal = response['subgoal']
        if "is complete" in subgoal or "is finished" in subgoal: # avoid using these subgoals as the final subgoal
            subgoal = last_subgoal
        return subgoal, False

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if not self.api:
            return
        self.api.save_conversation()
        self.api.prepare_input_data(
            epstate.image_buffer,
            self._get_text_query(epstate.count),
            epstate.count,
        )
        self.api.save_final_video(f"{success_flag}_ep{self.episode_id}_{self.task_goal}.mp4")
        self.api.clear_uploaded_files()
        del self.api

    def _get_text_query(self, count: int) -> str:
        if count == 0:
            if self.env_name in TASK_WITH_VIDEO_DEMO:
                template = DEMO_TEXT_QUERY
            else:
                template = IMAGE_TEXT_QUERY
        else:
            template = VIDEO_TEXT_QUERY
        return template.format(task_goal=self.task_goal)

    def _should_call(self, count: int) -> bool:
        if count == 0:
            return True
        if self.env_name in LONG_FIRST_ACTION_TASKS and count < 75:
            return False # avoid changing the first action too early
        return count % 48 == 0


class QwenVLSubgoalPredictor(SubgoalPredictorBase):
    
    def setup_api(self) -> None:
        self.api = Qwen3VLModel(
            adapter_path=self.args.qwenvl_simpleSG_adapter_path if self.args.subgoal_type == "simple_subgoal" else self.args.qwenvl_groundSG_adapter_path,
            subgoal_type=self.args.subgoal_type,
        )
        print(f"[robomme] QwenVL {self.args.subgoal_type} agent setup finished")
        
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        # Use a sub-folder so the `shutil.rmtree(self.episode_dir)` in
        # end_episode only wipes the predictor's per-step PNG dumps, not
        # the StructuredTraceRecorder's trace.json+frames/ that live at
        # {save_dir}/{task}/ep{N}/.
        self.episode_dir = os.path.join(
            self.save_dir, self.env_name, f"ep{self.episode_id}", "_predictor_workdir"
        )
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # Some tricks. QwenVL sometimes thinks the button has been pressed. hot fix for now.
        # Such special tricks are not encouraged if you consider participate RoboMME challenge @ CVPR 2026
        if self.env_name in ["ButtonUnmask", "PickHighlight"]:
            keep_period = 90
        elif self.env_name == "ButtonUnmaskSwap":
            if last_subgoal and "press the first button" in last_subgoal:
                keep_period = 100
            elif last_subgoal and "press the second button" in last_subgoal:
                keep_period = 250
            else:
                keep_period = 0
        else:
            keep_period = 0

        response = self.api.call(self.video_buffer[-1], count, keep_period)
        self.video_buffer.clear()
        return response, False

    def get_last_vlm_call(self) -> Optional[dict]:
        raw = getattr(self.api, "last_response", None)
        if raw is None:
            return None
        return {
            "raw_response": raw,
            "subgoal_clean": raw,  # SimpleSG/GroundSG: response IS the subgoal
        }

    def get_prompt_template(self) -> Optional[dict]:
        is_simple = self.args.subgoal_type == "simple_subgoal"
        subgoal_word = "language" if is_simple else "grounded language"
        system = (
            f"You are a helpful assistant to help guide the robot to complete the task "
            f"by predicting a sequence of {subgoal_word} subgoals"
        )
        user_initial = (
            "{video_prefix}The task goal is: {task_goal}\n"
            "This is the initial turn for prediction\n"
            f"<image>What's the next {subgoal_word} subgoal based on current observation?"
        )
        user_with_history = (
            "{video_prefix}The task goal is: {task_goal}\n"
            f"The history of previous predicted {subgoal_word} subgoals are: "
            "{history_subgoals}\n"
            f"<image>What's the next {subgoal_word} subgoal based on current observation?"
        )
        return {
            "system": system,
            "user_template_initial_turn": user_initial,
            "user_template_with_history": user_with_history,
            "expected_response_format": "<subgoal_text>",
            "predictor": "qwenvl",
            "subgoal_type": self.args.subgoal_type,
        }

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if self.episode_dir:
            shutil.rmtree(self.episode_dir) # save some space, you can comment this function out to keep all video frames


class MemERSubgoalPredictor(SubgoalPredictorBase):
    def setup_api(self) -> None:
        self.api = Qwen3VLModelMemER(adapter_path=self.args.memer_adapter_path)
        print("[robomme] MemER agent setup finished")
    
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.episode_dir = os.path.join(
            self.save_dir, self.env_name, f"ep{self.episode_id}", "_predictor_workdir"
        )
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.api.add_execution_frame(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        response = self.api.call()
        return response, False

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if self.episode_dir:
            shutil.rmtree(self.episode_dir) # save some space, you can comment this function out to keep all video frames


class MemorySubgoalPredictor(SubgoalPredictorBase):
    """Memory-augmented Qwen3-VL subgoal predictor (our v2 variants).

    `with_history` toggles between the two trained variants:
      * True  -> v2-history (history line + memory)
      * False -> v2-only    (memory only, no numbered history)
    """

    with_history: bool = True

    def setup_api(self) -> None:
        adapter_path = (
            self.args.memory_history_adapter_path
            if self.with_history
            else self.args.memory_only_adapter_path
        )
        # memory_style ∈ {simple, layout, grounded, dual} — must match what the
        # adapter was trained on (build_memory_subgoal_jsonl.py --memory-style).
        # For dual, text_style further selects which textual action memory was
        # paired with the spatial memory (--text-style at training time). The
        # API derives system_prompt + subgoal_type + wording from the pair.
        text_style = getattr(self.args, "text_style", "present")
        self.api = Qwen3VLModelMemory(
            adapter_path=adapter_path,
            with_history=self.with_history,
            memory_style=self.args.memory_style,
            text_style=text_style,
        )
        ts_blurb = (
            f", text_style={text_style}" if self.args.memory_style == "dual" else ""
        )
        print(
            f"[robomme] Memory ({'history' if self.with_history else 'only'}, "
            f"memory_style={self.args.memory_style}{ts_blurb}) agent setup finished"
        )

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.episode_dir = os.path.join(
            self.save_dir, self.env_name, f"ep{self.episode_id}", "_predictor_workdir"
        )
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # Snapshot pre-call memories so we can record m_t -> m_{t+1}.
        self._last_memory_in = self.api.current_memory
        self._last_spatial_memory_in = (
            self.api.current_spatial_memory if self.api.uses_spatial else None
        )
        response = self.api.call(self.video_buffer[-1], count, keep_period=0)
        self.video_buffer.clear()
        return response, False

    def get_last_vlm_call(self) -> Optional[dict]:
        raw = getattr(self.api, "last_response", None)
        if raw is None:
            return None
        # api._parse_memory_response splits subgoal vs action-memory; reuse it.
        subgoal_clean, mem_out = self.api._parse_memory_response(raw)
        info = {
            "raw_response": raw,
            "subgoal_clean": subgoal_clean,
            "memory_in": getattr(self, "_last_memory_in", None),
            "memory_out": mem_out,
            "with_history": self.with_history,
            "memory_style": self.api.memory_style,
        }
        if self.api.uses_spatial:
            info["spatial_memory_in"]  = getattr(self, "_last_spatial_memory_in", None)
            info["spatial_memory_out"] = self.api._parse_spatial_memory(raw)
        return info

    def get_prompt_template(self) -> Optional[dict]:
        # Build templates that mirror api_memory.Qwen3VLModelMemory._user_prompt
        # for the current memory_style.
        sg_word = self.api._sg_word
        uses_spatial = self.api.uses_spatial
        action_mem_line = (
            "The current action memory (a brief summary of relevant past events): "
            "<mem>{memory_in}</mem>"
        ) if uses_spatial else (
            "The current memory (a brief summary of relevant past events): "
            "<mem>{memory_in}</mem>"
        )
        spatial_mem_line = (
            "\nThe current spatial memory (a description of the current scene): "
            "<spatial_mem>{spatial_memory_in}</spatial_mem>"
        ) if uses_spatial else ""

        if self.with_history:
            user_initial = (
                "{video_prefix}The task goal is: {task_goal}\n"
                "This is the initial turn for prediction\n"
                + action_mem_line + spatial_mem_line + "\n"
                f"<image>What's the next {sg_word} subgoal based on current observation?"
            )
            user_with_history = (
                "{video_prefix}The task goal is: {task_goal}\n"
                f"The history of previous predicted {sg_word} subgoals are: "
                "{history_subgoals}\n"
                + action_mem_line + spatial_mem_line + "\n"
                f"<image>What's the next {sg_word} subgoal based on current observation?"
            )
        else:
            user_initial = (
                "{video_prefix}The task goal is: {task_goal}\n"
                + action_mem_line + spatial_mem_line + "\n"
                f"<image>What's the next {sg_word} subgoal based on current observation?"
            )
            user_with_history = user_initial
        expected_response = (
            "<subgoal_text>\\n<mem>{memory_out}</mem>\\n<spatial_mem>{spatial_memory_out}</spatial_mem>"
            if uses_spatial else
            "<subgoal_text>\\n<mem>{memory_out}</mem>"
        )
        return {
            "system": self.api.system_prompt,
            "user_template_initial_turn": user_initial,
            "user_template_with_history": user_with_history,
            "expected_response_format": expected_response,
            "predictor": "memory_history" if self.with_history else "memory_only",
            "with_history": self.with_history,
            "memory_style": self.args.memory_style,
            "subgoal_type": self.api.subgoal_type,
            "uses_spatial": uses_spatial,
            "initial_memory": "No actions had been completed yet.",
            "initial_spatial_memory": "No observations yet." if uses_spatial else None,
        }

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if self.episode_dir:
            shutil.rmtree(self.episode_dir)


class MemoryHistorySubgoalPredictor(MemorySubgoalPredictor):
    with_history = True


class MemoryOnlySubgoalPredictor(MemorySubgoalPredictor):
    with_history = False


class OracleSubgoalPredictor(SubgoalPredictorBase):

    def setup_api(self) -> None:
        print("[robomme] Oracle agent setup finished")

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if self.args.subgoal_type == "simple_subgoal":
            sg = self.env_runner.simple_subgoal_oracle
        else:
            sg = self.env_runner.grounded_subgoal_oracle
        self._last_oracle_subgoal = sg
        return sg, False

    def get_last_vlm_call(self) -> Optional[dict]:
        sg = getattr(self, "_last_oracle_subgoal", None)
        if sg is None:
            return None
        return {"raw_response": sg, "subgoal_clean": sg, "source": "oracle"}

    def get_prompt_template(self) -> Optional[dict]:
        return {
            "predictor": "oracle",
            "subgoal_type": self.args.subgoal_type,
            "note": (
                "Oracle: subgoals come directly from the sim "
                "(env_runner.{simple|grounded}_subgoal_oracle); no VLM prompt."
            ),
        }


def build_subgoal_predictor(
    args,
    save_dir: Path,
) -> SubgoalPredictorBase:
    if args.use_gemini:
        return GeminiSubgoalPredictor(args, save_dir)
    if args.use_qwenvl:
        return QwenVLSubgoalPredictor(args, save_dir)
    if args.use_memer:
        return MemERSubgoalPredictor(args, save_dir)
    if getattr(args, "use_memory_history", False):
        return MemoryHistorySubgoalPredictor(args, save_dir)
    if getattr(args, "use_memory_only", False):
        return MemoryOnlySubgoalPredictor(args, save_dir)
    if args.use_oracle:
        return OracleSubgoalPredictor(args, save_dir)
    
    return NullSubgoalPredictor(args, save_dir)



