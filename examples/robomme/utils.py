import os
import json
import numpy as np
import cv2
import imageio
import re
import collections
from pathlib import Path
from typing import Tuple, Optional, Any



TASK_WITH_VIDEO_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"
]

TASK_NAME_LIST=  [      
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick"
]

SUBGOAL_TYPES = ("simple_subgoal", "grounded_subgoal")



def pack_buffer(image_buffer, state_buffer, exec_start_idx=0):
    image_output = np.stack(image_buffer, axis=0).astype(np.uint8)[:, None]
    state_output = np.stack(state_buffer, axis=0).astype(np.float32)
    return {
        "images": image_output,
        "state": state_output,
        "add_buffer": True,
        "exec_start_idx": exec_start_idx,
    }
    
def check_args(args):
    assert args.subgoal_type in ["simple_subgoal", "grounded_subgoal", None] and args.obs_horizon == 16
    if args.use_memer:
        args.subgoal_type = "grounded_subgoal"



class EpisodeState:
    def __init__(self):
        self.image_buffer = []
        self.wrist_image_buffer = []
        self.state_buffer = []
        self.action_plan = collections.deque()
        self.count = 0
        self.exec_start_idx = 0

    def add_observation(self, img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray):
        self.image_buffer.append(img.copy())
        self.wrist_image_buffer.append(wrist_img.copy())
        self.state_buffer.append(state.copy())

    def clear_buffers(self):
        self.image_buffer.clear()
        self.wrist_image_buffer.clear()
        self.state_buffer.clear()
        self.exec_start_idx = 0

    def get_current_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_buffer[-1], self.wrist_image_buffer[-1], self.state_buffer[-1]





class RolloutRecorder:
    def __init__(self, save_dir: str, task_goal: str, fps: int = 30):
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self.total_images = []
        self.fps = fps
        self.task_goal = task_goal
        
    def _extract_points(self, subgoal: str):
        match = re.findall(r'<(\d+), (\d+)>', subgoal)
        points = []
        for m in match:
            points.append((int(m[0]), int(m[1])))
        return points
        
    # Hard cap on the subgoal string rendered into the rollout video overlay.
    # `add_text_area` word-wraps to whatever height the text needs, so a
    # 200+ char malformed-VLM-output subgoal grows the overlay past previous
    # frames' overlay heights → frames in `self.total_images` end up with
    # different shapes → imageio.mimsave fails with "All images in a movie
    # should have same size" at episode close. 120 chars fits the longest
    # realistic well-formed subgoal (~80 chars: "pick up the second
    # highlighted cube at <120, 88>") with headroom for the prefix.
    _SUBGOAL_RENDER_CAP_CHARS = 120

    def record(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, action: np.ndarray=None, is_video_demo: bool=False, subgoal: Optional[str] = None):

        concat_image = np.concatenate([image, wrist_image], axis=1)
        if is_video_demo: # add a red border
            concat_image = cv2.rectangle(concat_image, (0, 0), (concat_image.shape[1], concat_image.shape[0]), (255, 0, 0), 10)

        frame_text = "Frame: " + str(len(self.total_images))
        frame_text_area = self.add_text_area(frame_text, concat_image.shape)

        goal_text = "Task Goal: " + self.task_goal
        goal_text_area = self.add_text_area(goal_text, concat_image.shape)

        if subgoal is not None:
            # Truncate so the overlay height stays bounded — see comment on
            # _SUBGOAL_RENDER_CAP_CHARS above. Note: this only affects the
            # video overlay; the FULL subgoal is still passed to the VLA
            # and recorded in trace.json elsewhere.
            display_subgoal = subgoal
            if len(display_subgoal) > self._SUBGOAL_RENDER_CAP_CHARS:
                display_subgoal = display_subgoal[: self._SUBGOAL_RENDER_CAP_CHARS - 3] + "..."
            subgoal_text = "Subgoal: " + display_subgoal
            subgoal_text_area = self.add_text_area(subgoal_text, concat_image.shape)

            if self._extract_points(subgoal) is not None:
                for point in self._extract_points(subgoal):
                    concat_image = cv2.circle(concat_image, point[::-1], 5, (255, 255, 0), -1)

            concat_image = np.concatenate([subgoal_text_area, concat_image], axis=0)
        
        state_text = "State: " + ','.join([f"{i:.4f}" for i in state])
        state_text_area = self.add_text_area(state_text, concat_image.shape)
        
        action_text = 'Action: ' + ','.join([f"{i:.4f}" for i in action]) if action is not None else "Action:None"
        action_text_area = self.add_text_area(action_text, concat_image.shape)
        # Concatenate text area on top of image
        concat_image = np.concatenate([frame_text_area, goal_text_area, action_text_area, state_text_area, concat_image], axis=0)
        self.total_images.append(concat_image)
    
    def add_text_area(self, text: str, concat_image_shape: tuple):        
        # Calculate text wrapping
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        max_width = concat_image_shape[1] - 20  # Leave 10px margin on each side
        lines = []
        words = text.replace(',', ' ').split()
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + ' ' + word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # Add the last line
        
        # Create text area with dynamic height
        line_height = 20
        text_area_height = max(50, len(lines) * line_height + 10)
        text_area = np.zeros((text_area_height, concat_image_shape[1], 3), dtype=np.uint8)
        
        # Draw each line
        for i, line in enumerate(lines):
            y_position = 15 + i * line_height
            text_area = cv2.putText(text_area, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)
        
        return text_area
             
    def save_video(self, filename: str):
        imageio.mimsave(os.path.join(self.save_dir, filename), self.total_images, fps=self.fps)


def build_viewer_results(
    *,
    task: str,
    episode_id: int,
    task_goal: str,
    model_label: str,
    subgoal_source: str,
    outcome: str,
    steps: list,
    vlm_calls: list,
    policy_chunks: list,
    t_episode_start: Optional[float],
    t_episode_end: Optional[float],
    eval_run: str,
    difficulty: Optional[str] = None,
    seed: Optional[int] = None,
    dataset: Optional[str] = None,
    model_ckpt_id: Optional[int] = None,
    camera_views: str = "both",
    video_fps: int = 30,
) -> dict:
    """Build a viewer-shaped results dict from the structured trace data.

    Lives at module level so the post-hoc converter
    (project/viewer/scripts/convert_canonical_eval.py) can import the
    same logic via its own copy of this signature — keeping the two
    code paths from drifting.

    Timing precedence:
      * If `vlm_calls` is non-empty, memory_log uses real timestamps.
      * If `policy_chunks` is non-empty, policy_log uses real timestamps.
      * Otherwise we fall back to scanning `steps` for is_subgoal_call
        markers and zero-fill timing — same as the legacy behavior.
    """

    if vlm_calls:
        memory_log = []
        for i, v in enumerate(vlm_calls):
            phase = v.get("phase") or ("init" if i == 0 else "update")
            vlm = v.get("vlm_call") or {}
            mem_text = vlm.get("memory_out") or vlm.get("raw_response") or ""
            spatial_out = vlm.get("spatial_memory_out")
            if spatial_out:
                mem_text = f"<mem>{mem_text}</mem>\n<spatial_mem>{spatial_out}</spatial_mem>"
            entry: dict[str, Any] = {
                "step": v.get("step", 0),
                "phase": phase,
                "global_time_start": v.get("global_time_start", 0.0),
                "global_time_end": v.get("global_time_end", 0.0),
                "time_s": v.get("time_s", 0.0),
                "memory": mem_text,
                "subtask": v.get("subtask"),
                "prompt_to_policy": v.get("subtask"),
            }
            if v.get("front_image"):
                entry["context_frame_indices"] = [v.get("step", 0)]
                entry["context_frame_paths"] = [v["front_image"]]
            memory_log.append(entry)
    else:
        memory_log = []
        first_seen = False
        for s in steps:
            if not s.get("is_subgoal_call"):
                continue
            vlm = s.get("vlm_call") or {}
            mem_text = vlm.get("memory_out") or vlm.get("raw_response") or ""
            spatial_out = vlm.get("spatial_memory_out")
            if spatial_out:
                mem_text = f"<mem>{mem_text}</mem>\n<spatial_mem>{spatial_out}</spatial_mem>"
            subtask = vlm.get("subgoal_clean") or s.get("subgoal")
            entry = {
                "step": s.get("step", 0),
                "phase": "init" if not first_seen else "update",
                "global_time_start": 0.0,
                "global_time_end": 0.0,
                "time_s": 0.0,
                "memory": mem_text,
                "subtask": subtask,
                "prompt_to_policy": subtask,
            }
            if s.get("front_image"):
                entry["context_frame_indices"] = [s.get("step", 0)]
                entry["context_frame_paths"] = [s["front_image"]]
            memory_log.append(entry)
            first_seen = True

    if policy_chunks:
        policy_log = [
            {
                "index": c.get("index", i),
                "step_start": c.get("step_start", 0),
                "step_end": c.get("step_end", 0),
                "infer_global_start": c.get("infer_global_start", 0.0),
                "infer_global_end": c.get("infer_global_end", 0.0),
                "env_global_start": c.get("env_global_start", 0.0),
                "env_global_end": c.get("env_global_end", 0.0),
                "infer_time_s": c.get("infer_time_s", 0.0),
                "env_time_s": c.get("env_time_s", 0.0),
                "subtask": c.get("subtask"),
                "prompt_to_policy": c.get("subtask"),
            }
            for i, c in enumerate(policy_chunks)
        ]
    else:
        policy_log = [
            {
                "index": i,
                "step_start": s.get("step", i),
                "step_end": s.get("step", i) + 1,
                "infer_global_start": 0.0, "infer_global_end": 0.0,
                "env_global_start": 0.0, "env_global_end": 0.0,
                "infer_time_s": 0.0, "env_time_s": 0.0,
                "subtask": s.get("subgoal"),
                "prompt_to_policy": s.get("subgoal"),
            }
            for i, s in enumerate(steps)
        ]

    vlm_timings = [v.get("time_s", 0.0) for v in vlm_calls]
    policy_timings = [c.get("infer_time_s", 0.0) for c in policy_chunks]
    avg_vlm = sum(vlm_timings) / len(vlm_timings) if vlm_timings else 0.0
    total_vlm = sum(vlm_timings)
    if t_episode_start is not None and t_episode_end is not None:
        total_rollout_s = max(0.0, t_episode_end - t_episode_start)
    elif policy_chunks:
        total_rollout_s = max(
            (c.get("env_global_end", 0.0) for c in policy_chunks),
            default=0.0,
        ) - (vlm_calls[0]["global_time_start"] if vlm_calls else policy_chunks[0]["infer_global_start"])
        total_rollout_s = max(0.0, total_rollout_s)
    else:
        total_rollout_s = 0.0

    video_frame_map = [
        {
            "frame_idx": i,
            "step": s.get("step", i),
            "type": "vlm_overlay" if s.get("is_subgoal_call") else "obs",
        }
        for i, s in enumerate(steps)
    ]

    return {
        "task_name": task,
        "task_goal": task_goal,
        "episode": episode_id,
        "seed": seed,
        "difficulty": difficulty,
        "dataset": dataset,
        "camera_views": camera_views,
        "outcome": outcome,
        "success": outcome == "success",
        "steps": len(steps),
        "task_description": task_goal,
        "memory_log": memory_log,
        "vlm_timings": vlm_timings,
        "policy_timings": policy_timings,
        "policy_log": policy_log,
        "avg_vlm_time_s": avg_vlm,
        "total_vlm_time_s": total_vlm,
        "total_rollout_time_s": total_rollout_s,
        "video_fps": video_fps,
        "video_frame_map": video_frame_map,
        "demo_num_frames": None,
        "demo_keyframe_indices": [],
        "demo_keyframe_paths": [],
        "eval_run": eval_run,
        "config": {
            "vlm_call_interval": None,
            "context_frames": 1,
            "frame_spacing": 1,
            "episodes_per_difficulty": None,
            "camera_views": camera_views,
            "dataset": dataset,
            "model": model_label,
            "model_ckpt_id": model_ckpt_id,
            "model_seed": seed,
            "subgoal_source": subgoal_source,
            "quantization": "lora-bf16",
            "gpu_setup": "2x A100-80GB: VLM on GPU 1, pi0.5 JAX on GPU 0",
        },
    }


class StructuredTraceRecorder:
    """Per-episode JSON + frames recorder for eval rollouts.

    Mirrors the structure of `mem_traces/<task>/ep<N>/trace.json` from
    `project/modal_mem_extract.py`, but populated live during eval rather
    than from a recorded oracle trajectory:

      trace.json:
        task, episode_id, task_goal, model_label, subgoal_source, outcome
        steps: [
          {step, state, action, subgoal,
           is_subgoal_call, vlm_call: {raw_response, input_image, ...}}, ...
        ]

    Image frames go to ./frames/{step:05d}_front.png + _wrist.png; only
    paths (relative) are stored in the JSON to keep it small.
    """

    def __init__(
        self,
        save_dir: Path,
        task: str,
        episode_id: int,
        task_goal: str,
        model_label: str,
        subgoal_source: str,
        prompt_template: Optional[dict] = None,
        save_every: int = 1,
    ):
        self.dir = Path(save_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        self.task = task
        self.episode_id = episode_id
        self.task_goal = task_goal
        self.model_label = model_label
        self.subgoal_source = subgoal_source
        self.prompt_template = prompt_template
        self.save_every = max(1, save_every)
        self.steps: list = []
        # Timing buckets. All times are wall-clock `time.time()` seconds
        # (UNIX epoch). The viewer's Gantt chart normalizes off the first
        # VLM start, so what matters is that all timestamps share a clock.
        # See InferenceTimingChart.tsx and project/viewer/src/types.ts.
        self.vlm_calls: list[dict] = []
        self.policy_chunks: list[dict] = []
        self.t_episode_start: Optional[float] = None
        self.t_episode_end: Optional[float] = None

    def record_step(
        self,
        step_idx: int,
        state: np.ndarray,
        action: Optional[np.ndarray],
        image: np.ndarray,
        wrist_image: np.ndarray,
        subgoal: Optional[str],
        is_subgoal_call: bool,
        vlm_call: Optional[dict] = None,
    ) -> None:
        rec: dict[str, Any] = {
            "step": step_idx,
            "state": state.tolist() if state is not None else None,
            "action": action.tolist() if action is not None else None,
            "subgoal": subgoal,
            "is_subgoal_call": is_subgoal_call,
        }
        if vlm_call is not None:
            rec["vlm_call"] = vlm_call
        if step_idx % self.save_every == 0:
            front_path = self.dir / "frames" / f"{step_idx:05d}_front.png"
            wrist_path = self.dir / "frames" / f"{step_idx:05d}_wrist.png"
            imageio.imwrite(front_path, image)
            imageio.imwrite(wrist_path, wrist_image)
            rec["front_image"] = front_path.relative_to(self.dir).as_posix()
            rec["wrist_image"] = wrist_path.relative_to(self.dir).as_posix()
        self.steps.append(rec)

    def mark_episode_start(self, t: float) -> None:
        self.t_episode_start = t

    def mark_episode_end(self, t: float) -> None:
        self.t_episode_end = t

    def record_vlm_call(
        self,
        *,
        step: int,
        t0: float,
        t1: float,
        subtask: Optional[str],
        vlm_call: Optional[dict] = None,
        phase: str = "update",
        front_image: Optional[str] = None,
    ) -> None:
        """One entry per VLM forward. Drives the Gantt chart's VLM row
        and the memory_log[] array in the viewer schema."""
        self.vlm_calls.append({
            "step": step,
            "global_time_start": t0,
            "global_time_end": t1,
            "time_s": t1 - t0,
            "subtask": subtask,
            "phase": phase,
            "vlm_call": vlm_call,
            "front_image": front_image,
        })

    def record_policy_chunk(
        self,
        *,
        index: int,
        step_start: int,
        step_end: int,
        infer_t0: float,
        infer_t1: float,
        env_t0: float,
        env_t1: float,
        subtask: Optional[str],
    ) -> None:
        """One entry per pi0.5 action-chunk fetch + its sim execution.
        Drives the Gantt chart's Policy and Sim rows."""
        self.policy_chunks.append({
            "index": index,
            "step_start": step_start,
            "step_end": step_end,
            "infer_global_start": infer_t0,
            "infer_global_end": infer_t1,
            "env_global_start": env_t0,
            "env_global_end": env_t1,
            "infer_time_s": infer_t1 - infer_t0,
            "env_time_s": env_t1 - env_t0,
            "subtask": subtask,
        })

    def finish(self, outcome: str) -> Path:
        trace = {
            "task": self.task,
            "episode_id": self.episode_id,
            "task_goal": self.task_goal,
            "model_label": self.model_label,
            "subgoal_source": self.subgoal_source,
            "prompt_template": self.prompt_template,
            "outcome": outcome,
            "n_steps": len(self.steps),
            "t_episode_start": self.t_episode_start,
            "t_episode_end": self.t_episode_end,
            "vlm_calls": self.vlm_calls,
            "policy_chunks": self.policy_chunks,
            "steps": self.steps,
        }
        out = self.dir / "trace.json"
        with open(out, "w") as f:
            json.dump(trace, f, indent=2)
        return out

    def write_viewer_results(
        self,
        outcome: str,
        video_path: Optional[Path] = None,
        eval_run: Optional[str] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
        dataset: Optional[str] = None,
        model_ckpt_id: Optional[int] = None,
        camera_views: str = "both",
        video_fps: int = 30,
    ) -> Path:
        """Emit a `results.json` shaped for the rollout viewer
        (`project/viewer/src/types.ts::EpisodeResults`) plus a `rollout.mp4`
        symlink next to it. Called from eval.py after `finish()` and after
        the .mp4 is saved, so we can keep the viewer payload in sync with
        the eval-time data.

        Schema mirrored by project/viewer/scripts/convert_canonical_eval.py.
        Keep the two in step if you change one."""
        results = build_viewer_results(
            task=self.task,
            episode_id=self.episode_id,
            task_goal=self.task_goal,
            model_label=self.model_label,
            subgoal_source=self.subgoal_source,
            outcome=outcome,
            steps=self.steps,
            vlm_calls=self.vlm_calls,
            policy_chunks=self.policy_chunks,
            t_episode_start=self.t_episode_start,
            t_episode_end=self.t_episode_end,
            eval_run=eval_run or self.model_label,
            difficulty=difficulty,
            seed=seed,
            dataset=dataset,
            model_ckpt_id=model_ckpt_id,
            camera_views=camera_views,
            video_fps=video_fps,
        )
        out = self.dir / "results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

        if video_path is not None and video_path.exists():
            dst = self.dir / "rollout.mp4"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            rel = os.path.relpath(video_path.resolve(), self.dir.resolve())
            try:
                os.symlink(rel, dst)
            except OSError:
                # Some filesystems (older fuse mounts on Modal volumes) reject
                # symlinks — fall back to a hardlink, then to a copy.
                try:
                    os.link(video_path, dst)
                except OSError:
                    import shutil as _shutil
                    _shutil.copy2(video_path, dst)
        return out