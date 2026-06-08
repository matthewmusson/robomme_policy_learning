import dataclasses
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Any, Tuple

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy
from utils import (
    pack_buffer,
    check_args,
    TASK_NAME_LIST,
    TASK_WITH_VIDEO_DEMO,
    SUBGOAL_TYPES,
    EpisodeState,
)
from utils import RolloutRecorder, StructuredTraceRecorder
from env_runner import EnvRunner
from subgoal_predictor import build_subgoal_predictor, SubgoalPredictorBase

# qwen3-vl environment variables
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'



@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8011

    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation"
    overwrite: bool = False

    use_history: bool = True
    policy_name: str = "dummy_test"
    model_seed: int = 42
    model_ckpt_id: int = 80000

    # task control
    re_eval_tasks: str = "" # tasks split by comma
    only_tasks: str = "" # tasks split by comma
    exclude_tasks: str = "" # tasks split by comma
    max_episodes_per_task: int = 0 # 0 means no cap; >0 limits each task to the first N episodes (used for smoke runs)
    episode_start: int = 0 # inclusive: only run episodes >= this index (per task). used for fan-out across parallel workers. SUPERSEDED by episode_ids_json when that's non-empty.
    episode_end: int = -1  # exclusive: only run episodes < this index (per task). -1 means "to the end". used for fan-out across parallel workers. SUPERSEDED by episode_ids_json when that's non-empty.
    episode_ids_json: str = ""  # JSON map `{"<task_name>": [<ep_id>, ...]}` of episodes to run, per task. When non-empty, supersedes episode_start/end — used by run_split_episodes to distribute pending eps EVENLY across workers (the static slice was unbalanced when most eps were already done from a prior run).
    save_traces: bool = True # write a per-episode trace.json (state/action/subgoal/vlm_call) + frames PNGs alongside the video
    trace_frame_every: int = 4 # only save every Nth frame as PNG to keep disk usage down (1 = every frame)
    consolidated_layout: bool = True # NEW LAYOUT: per-task folders, per-episode result.json (no progress.json race across parallel workers). When True, save_dir contents look like: {save_dir}/{task}/ep{N}.result.json, ep{N}.mp4, ep{N}/trace.json+frames/. When False, the legacy nested layout (policy_name/ckpt/seed/predictor/videos+traces+progress.json+log.json) is used.

    # VLM subgoal predictor
    use_oracle: bool = False
    use_qwenvl: bool = False
    use_memer: bool = False
    use_gemini: bool = False
    use_memory_history: bool = False   # our v2-history variant
    use_memory_only: bool = False      # our v2-only variant
    subgoal_type: Optional[str] = None  # [simple_subgoal, grounded_subgoal]
    gemini_model_name: str = "gemini-2.5-pro"
    qwenvl_simpleSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/simple_subgoal/checkpoint-1400"
    qwenvl_groundSG_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200"
    memer_adapter_path: str = "runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300"
    memory_history_adapter_path: str = ""   # our v2-history LoRA checkpoint
    memory_only_adapter_path: str = ""      # our v2-only LoRA checkpoint
    memory_style: str = "simple"            # {simple, layout, grounded, dual}; only used by memory predictors. Must match what the adapter was trained on. Also auto-derives subgoal_type if subgoal_type was left blank.
    text_style: str = "present"             # only meaningful when memory_style=="dual". {simple, present, grounded, layout}. Picks the textual action memory variant the dual adapter was trained against. "present" preserves the legacy dual behavior.
    subgoal_keep_period: int = 1 # ever subgoal should be kept for this many steps
    # this can accelerate the evaluation process for symbolic memory
    # In our experiments, we just set this to 1



def _validate_task_names(names, arg: str) -> None:
    """Raise with a clear, suggestion-augmented message if any name in
    `names` isn't in TASK_NAME_LIST. Catches case typos (`PickXTimes` →
    suggest `PickXtimes`) and stray whitespace before they reach
    EnvRunner and crash a worker mid-run.

    `arg` is the user-facing flag name to cite in the error (e.g.
    `--args.only-tasks`)."""
    import difflib
    canonical = set(TASK_NAME_LIST)
    bad = [n for n in names if n not in canonical]
    if not bad:
        return
    suggestions = []
    for n in bad:
        close = difflib.get_close_matches(n, TASK_NAME_LIST, n=1, cutoff=0.6)
        if close:
            suggestions.append(f"  {n!r} — did you mean {close[0]!r}?")
        else:
            suggestions.append(f"  {n!r} — no close match; valid: {TASK_NAME_LIST}")
    raise ValueError(
        f"{arg} contains {len(bad)} task name(s) that aren't in TASK_NAME_LIST:\n"
        + "\n".join(suggestions)
    )


def _resolve_subgoal_source(args: "Args") -> str:
    """String label of which predictor branch is active, for trace metadata."""
    if args.use_gemini: return "gemini"
    if args.use_qwenvl: return "qwenvl"
    if args.use_memer: return "memer"
    if args.use_memory_history: return "memory_history"
    if args.use_memory_only: return "memory_only"
    if args.use_oracle: return "oracle"
    return "none"


class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        subgoal_predictor: SubgoalPredictorBase,
        video_save_dir: Path,
    ) -> str:
        client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(
            self.args.host, self.args.port
        )
        resp = client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        epstate = EpisodeState()
        task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)
        subgoal_predictor.start_episode(epstate, env_runner)

        trace_recorder = None
        if self.args.save_traces:
            if self.args.consolidated_layout:
                # New layout: {save_dir}/{task}/ep{N}/
                trace_dir = self.save_dir / env_runner.env_id / f"ep{env_runner.episode_id}"
            else:
                # Legacy layout: {save_dir}/traces/{task}/ep{N}/
                trace_dir = self.save_dir / "traces" / env_runner.env_id / f"ep{env_runner.episode_id}"
            trace_recorder = StructuredTraceRecorder(
                save_dir=trace_dir,
                task=env_runner.env_id,
                episode_id=env_runner.episode_id,
                task_goal=task_goal,
                model_label=self.args.policy_name,
                subgoal_source=_resolve_subgoal_source(self.args),
                prompt_template=subgoal_predictor.get_prompt_template(),
                save_every=self.args.trace_frame_every,
            )

        img, wrist_img, robot_state = epstate.get_current_obs()
        prompt = task_goal
        success_flag = "unknown"
        subgoal = None
        last_subgoal = None
        is_subgoal_call_this_step = False

        # Timing instrumentation for the Gantt chart in
        # project/viewer/src/components/InferenceTimingChart.tsx. We record:
        #   * One vlm_calls entry per `get_subgoal()` call (Qwen forward).
        #   * One policy_chunks entry per `get_action_chunk()` call + the
        #     sim window that consumes the chunk.
        # All timestamps are `time.time()` (a common wall-clock); the viewer
        # normalizes off the first VLM start.
        t_episode_start = time.time()
        if trace_recorder is not None:
            trace_recorder.mark_episode_start(t_episode_start)
        chunk_open = False
        chunk_index = 0
        chunk_step_start = 0
        chunk_infer_t0 = chunk_infer_t1 = 0.0
        chunk_env_t0 = 0.0
        chunk_subtask: Optional[str] = None
        last_env_t1 = t_episode_start

        while True:
            subgoal_predictor.step(epstate)
            is_subgoal_call_this_step = False

            if not epstate.action_plan:
                # Close out the previous chunk (its sim segment ended at the
                # last env.step) before starting a new one.
                if chunk_open and trace_recorder is not None:
                    trace_recorder.record_policy_chunk(
                        index=chunk_index,
                        step_start=chunk_step_start,
                        step_end=epstate.count,
                        infer_t0=chunk_infer_t0,
                        infer_t1=chunk_infer_t1,
                        env_t0=chunk_env_t0,
                        env_t1=last_env_t1,
                        subtask=chunk_subtask,
                    )
                    chunk_index += 1
                    chunk_open = False

                vlm_t0 = vlm_t1 = None
                if epstate.count % self.args.subgoal_keep_period == 0 or last_subgoal is None:
                    vlm_t0 = time.time()
                    subgoal, has_api_error = subgoal_predictor.get_subgoal(
                        epstate.count,
                        subgoal,
                        last_subgoal,
                    )
                    vlm_t1 = time.time()
                    is_subgoal_call_this_step = True
                    if trace_recorder is not None:
                        trace_recorder.record_vlm_call(
                            step=epstate.count,
                            t0=vlm_t0,
                            t1=vlm_t1,
                            subtask=subgoal,
                            vlm_call=subgoal_predictor.get_last_vlm_call(),
                            phase="init" if chunk_index == 0 else "update",
                        )
                else:
                    subgoal = last_subgoal
                    has_api_error = False

                if has_api_error:
                    break

                infer_t0 = time.time()
                action_chunk = self.get_action_chunk(
                    client, epstate, img, wrist_img, robot_state, prompt, subgoal,
                    exec_horizon=self.args.obs_horizon
                )
                infer_t1 = time.time()

                epstate.action_plan.extend(action_chunk)
                epstate.clear_buffers()

                last_subgoal = subgoal

                # Open a fresh chunk record; env starts immediately after infer.
                chunk_open = True
                chunk_step_start = epstate.count
                chunk_infer_t0 = infer_t0
                chunk_infer_t1 = infer_t1
                chunk_env_t0 = infer_t1
                chunk_subtask = subgoal

            action = epstate.action_plan.popleft()
            obs, stop_flag, success_flag = env_runner.step(action)
            epstate.count += 1
            last_env_t1 = time.time()

            if epstate.count > self.args.max_steps:
                success_flag = "timeout"
                break

            img, wrist_img, robot_state = obs

            epstate.add_observation(img, wrist_img, robot_state)
            recorder.record(
                image=img.copy(),
                wrist_image=wrist_img.copy(),
                state=robot_state.copy(),
                action=action.copy(),
                subgoal=subgoal,
            )
            if trace_recorder is not None:
                trace_recorder.record_step(
                    step_idx=epstate.count,
                    state=robot_state,
                    action=action,
                    image=img,
                    wrist_image=wrist_img,
                    subgoal=subgoal,
                    is_subgoal_call=is_subgoal_call_this_step,
                    vlm_call=(
                        subgoal_predictor.get_last_vlm_call()
                        if is_subgoal_call_this_step else None
                    ),
                )

            if stop_flag:
                break

        # Final chunk close (loop broke out mid-execution: timeout / stop / error).
        if chunk_open and trace_recorder is not None:
            trace_recorder.record_policy_chunk(
                index=chunk_index,
                step_start=chunk_step_start,
                step_end=epstate.count,
                infer_t0=chunk_infer_t0,
                infer_t1=chunk_infer_t1,
                env_t0=chunk_env_t0,
                env_t1=last_env_t1,
                subtask=chunk_subtask,
            )

        if trace_recorder is not None:
            trace_recorder.mark_episode_end(time.time())

        if success_flag == "unknown":
            return "unknown"

        video_filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
        recorder.save_video(video_filename)

        if trace_recorder is not None:
            trace_recorder.finish(success_flag)
            # Also tee a viewer-shaped results.json + rollout.mp4 symlink so
            # the rollout viewer (project/viewer/) can ingest these rollouts
            # without a post-hoc conversion step. See StructuredTraceRecorder
            # docstring for the schema; convert_canonical_eval.py mirrors it
            # for older runs that predate this codepath.
            trace_recorder.write_viewer_results(
                outcome=success_flag,
                video_path=Path(video_save_dir) / video_filename,
                eval_run=self.args.policy_name,
                difficulty=getattr(env_runner, "difficulty", None),
                seed=self.args.model_seed,
                model_ckpt_id=self.args.model_ckpt_id,
            )

        subgoal_predictor.end_episode(epstate, success_flag)
        return success_flag


    def init_episode(
        self,
        env_runner: EnvRunner,
        epstate: EpisodeState,
        video_save_dir: Path,
    ) -> Tuple[str, RolloutRecorder]:
        pre_traj = env_runner.get_init_obs()
        task_goal = pre_traj["task_goal"]

        recorder = RolloutRecorder(video_save_dir, task_goal, fps=30)

        print(f"task_goal: {task_goal}")

        epstate.image_buffer.extend(pre_traj["images"])
        epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
        epstate.state_buffer.extend(pre_traj["states"])

        for i in range(len(pre_traj["images"])):
            recorder.record(
                image=pre_traj["images"][i].copy(),
                wrist_image=pre_traj["wrist_images"][i].copy(),
                state=pre_traj["states"][i].copy(),
                is_video_demo=env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1,
                subgoal=None if self.args.subgoal_type is None else "[initializing...]",
            )

        epstate.exec_start_idx = len(epstate.image_buffer) - 1
        print(f"exec_start_idx: {epstate.exec_start_idx}")
        return task_goal, recorder

    def get_action_chunk(
        self,
        client,
        state: EpisodeState,
        img: np.ndarray,
        wrist_img: np.ndarray,
        robot_state: np.ndarray,
        prompt: str,
        subgoal: Optional[str],
        exec_horizon: int,
    ) -> list:
        if self.args.use_history:
            resp = client.add_buffer(pack_buffer(
                state.image_buffer,
                state.state_buffer,
                state.exec_start_idx,
            ))
            while not resp.get("add_buffer_finished", False):
                time.sleep(0.1)

        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": robot_state,
            "prompt": prompt,
        }

        if subgoal is not None:
            element['simple_subgoal'] = subgoal
            element['grounded_subgoal'] = subgoal

        action_chunk = client.infer(element)["actions"]
        return action_chunk[:exec_horizon]


def setup_save_directory(args: Args) -> Path:
    """Set up and validate save directories.

    consolidated_layout=True: save_dir is used as-is (e.g.
        /runs/eval_canonical/<label>/), and per-task folders are created
        underneath at runtime.

    consolidated_layout=False (legacy): save_dir is nested under
        <save_dir>/<policy_name>/ckpt<X>/seed<Y>/<predictor>/.
    """
    if args.consolidated_layout:
        save_dir = Path(args.save_dir)
    else:
        save_dir = (
            Path(args.save_dir)
            / args.policy_name
            / f"ckpt{args.model_ckpt_id}"
            / f"seed{args.model_seed}"
        )
        if args.subgoal_type in SUBGOAL_TYPES:
            if args.use_gemini:
                save_dir = save_dir / "gemini"
            elif args.use_qwenvl:
                save_dir = save_dir / "qwenvl"
            elif args.use_memer:
                save_dir = save_dir / "memer"
            elif args.use_memory_history:
                save_dir = save_dir / "memory_history"
            elif args.use_memory_only:
                save_dir = save_dir / "memory_only"
            else:
                save_dir = save_dir / "oracle"

    if save_dir.exists() and not args.consolidated_layout:
        # Don't wipe in consolidated mode — multiple parallel workers share
        # this dir, and "resume" is handled per-episode via *.result.json
        # existence checks instead of a single progress.json.
        if args.overwrite:
            shutil.rmtree(save_dir)
            print(f"we will overwrite the evaluation at {save_dir}")
        else:
            print("we will resume the evaluation")

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def setup_log_dict(save_dir: Path, args: Args) -> dict:
    if os.path.exists(save_dir / "progress.json"):
        with open(save_dir / "progress.json", "r") as f:
            log_dict = json.load(f)

    elif os.path.exists(save_dir / "log.json"):
        with open(save_dir / "log.json", "r") as f:
            log_dict = json.load(f)
        log_dict.pop("success_rate", None)
        log_dict.pop("total_success_rate", None)
    else:
        log_dict = {}

    for task_name in log_dict:
        error_list = []
        for k, v in log_dict[task_name].items():
            if v == "error":
                error_list.append(k)
        for k in error_list:
            log_dict[task_name].pop(k)

    if args.re_eval_tasks:
        for task_name in args.re_eval_tasks.split(","):
            if task_name in log_dict:
                del log_dict[task_name]
                os.system(f"rm -f {save_dir / 'videos' / f'{task_name}_ep*.mp4'}")

    with open(save_dir / "progress.json", "w") as f:
        json.dump(log_dict, f, indent=2)

    return log_dict


def evaluate(args: Args):
    """Main evaluation function."""
    # When using a memory predictor, derive subgoal_type from memory_style
    # if the caller didn't pass one. layout and grounded both target
    # grounded subgoals; simple targets simple subgoals.
    if (args.use_memory_history or args.use_memory_only) and not args.subgoal_type:
        args.subgoal_type = (
            "simple_subgoal" if args.memory_style == "simple"
            else "grounded_subgoal"
        )

    check_args(args)

    save_dir = setup_save_directory(args)

    # Strip whitespace + drop empties so a shell-friendly
    # "BinFill, ButtonUnmask," doesn't blow up downstream
    # (env_runner compares strings literally against TASK_NAME_LIST).
    if args.only_tasks:
        task_names = [t.strip() for t in args.only_tasks.split(",") if t.strip()]
    else:
        task_names = TASK_NAME_LIST
    if args.exclude_tasks:
        excluded_raw = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()]
        _validate_task_names(excluded_raw, arg="--args.exclude-tasks")
        excluded = set(excluded_raw)
        task_names = [t for t in task_names if t not in excluded]

    # Fail fast on typos — silently dropping bad names looks like a
    # successful run with mysterious coverage gaps, and crashing inside
    # EnvRunner kills workers mid-loop after wasting model-load time.
    _validate_task_names(task_names, arg="--args.only-tasks")

    subgoal_predictor = build_subgoal_predictor(args, save_dir)
    evaluator = EpisodeEvaluator(args, save_dir)

    if args.consolidated_layout:
        _evaluate_consolidated(args, save_dir, task_names, subgoal_predictor, evaluator)
    else:
        _evaluate_legacy(args, save_dir, task_names, subgoal_predictor, evaluator)


def _evaluate_consolidated(args, save_dir, task_names, subgoal_predictor, evaluator):
    """Per-task folders, per-episode result.json. No global progress.json
    write contention across workers — each episode writes its own file.
    Aggregation into log.json is left to the orchestrator that spawned
    the workers.

    Semantics for `--max_episodes_per_task=N`: run up to N NEW episodes per
    task — i.e. N episodes that don't yet have a `ep<N>.result.json` on
    disk. This means re-invoking with the same N keeps appending fresh
    rollouts until the simulator's pool is exhausted (typically 50 eps).
    The previous behavior (cap absolute ep index range) was awkward
    because once the first N were done, re-running was a no-op.

    `--episode_start/end` still defines the slice this worker is allowed
    to draw from (used by run_split_episodes for fan-out). N is the cap
    within that slice, not on top of all done eps elsewhere.
    """
    # Parse the per-worker episode-id map from the orchestrator, if provided.
    # When set, this OVERRIDES the [episode_start, episode_end) slice — each
    # worker gets an explicit list of which ep_ids to run per task, chosen by
    # the orchestrator to balance pending work across workers.
    ep_id_whitelist: dict = {}
    if args.episode_ids_json:
        try:
            ep_id_whitelist = json.loads(args.episode_ids_json)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--args.episode-ids-json must be valid JSON, got: "
                f"{args.episode_ids_json[:200]!r}"
            ) from e
        if not isinstance(ep_id_whitelist, dict):
            raise ValueError(
                f"--args.episode-ids-json must decode to a dict, got: {type(ep_id_whitelist).__name__}"
            )

    for task_name in task_names:
        task_dir = save_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        env_runner = EnvRunner(task_name, task_dir, max_steps=args.max_steps)
        pool_size = env_runner.num_episodes

        # Decide which ep_ids this worker should attempt.
        if ep_id_whitelist:
            candidate_ids = sorted(int(x) for x in ep_id_whitelist.get(task_name, []))
            slice_descr = f"explicit-list len={len(candidate_ids)}"
        else:
            ep_start = max(0, args.episode_start)
            ep_end = pool_size if args.episode_end < 0 else min(pool_size, args.episode_end)
            candidate_ids = list(range(ep_start, ep_end))
            slice_descr = f"slice=[{ep_start},{ep_end})"

        # Scan disk once to find already-completed episodes
        # (atomic write of ep<N>.result.json gates a "done" episode).
        done_ids = set()
        for rp in task_dir.glob("ep*.result.json"):
            try:
                done_ids.add(int(rp.name[len("ep"):-len(".result.json")]))
            except ValueError:
                continue

        pending = [i for i in candidate_ids if i not in done_ids]
        cap = args.max_episodes_per_task if args.max_episodes_per_task and args.max_episodes_per_task > 0 else len(pending)
        to_run = pending[:cap]

        print(
            f"\n[robomme] {task_name}: pool={pool_size}, {slice_descr}, "
            f"done={len(done_ids)}, pending={len(pending)}, cap={cap} -> running {len(to_run)} "
            f"ep ids={to_run if len(to_run) <= 12 else to_run[:6] + ['...'] + to_run[-6:]}"
        )

        for episode_id in to_run:
            result_path = task_dir / f"ep{episode_id}.result.json"
            if result_path.exists():
                # Race against a parallel worker that grabbed the same ep.
                print(f"[robomme] {task_name}/ep{episode_id} appeared mid-loop, skipping")
                continue

            env_runner.make_env(episode_id)
            print(f"\n[robomme] env for {task_name}/ep{episode_id} setup finished")
            try:
                success_flag = evaluator.eval_each_episode(
                    env_runner, subgoal_predictor, task_dir
                )
                outcome = "error" if success_flag == "unknown" else success_flag
            except Exception as e:
                print(f"Error evaluating {task_name}/ep{episode_id}: {e}")
                outcome = "error"

            env_runner.close_env()
            result = {
                "task": task_name,
                "episode_id": episode_id,
                "outcome": outcome,
                "success": outcome == "success",
                "model_label": args.policy_name,
                "model_ckpt_id": args.model_ckpt_id,
                "model_seed": args.model_seed,
                "subgoal_source": _resolve_subgoal_source(args),
            }
            # write atomically via temp + rename, so concurrent readers
            # (e.g. an orchestrator polling for results) never see a partial file
            tmp = result_path.with_suffix(".result.json.tmp")
            with open(tmp, "w") as f:
                json.dump(result, f, indent=2)
            os.replace(tmp, result_path)

        del env_runner
        time.sleep(1)


def _evaluate_legacy(args, save_dir, task_names, subgoal_predictor, evaluator):
    """Original RoboMME layout: progress.json + log.json at save_dir top,
    videos/ flat. Kept for backward compatibility with compute_results.py."""
    video_save_dir = save_dir / "videos"
    log_dict = setup_log_dict(save_dir, args)
    if args.exclude_tasks:
        for task in args.exclude_tasks.split(","):
            task = task.strip()
            if not task:
                continue
            log_dict[task] = {str(i): False for i in range(50)}

    while not os.path.exists(save_dir / "log.json"):
        for task_name in task_names:
            if task_name not in log_dict:
                log_dict[task_name] = {}
            env_runner = EnvRunner(task_name, video_save_dir, max_steps=args.max_steps)
            num_episodes = env_runner.num_episodes
            if args.max_episodes_per_task and args.max_episodes_per_task > 0:
                num_episodes = min(num_episodes, args.max_episodes_per_task)
            ep_start = max(0, args.episode_start)
            ep_end = num_episodes if args.episode_end < 0 else min(num_episodes, args.episode_end)

            success_flag = "unknown"
            for episode_id in range(ep_start, ep_end):
                if str(episode_id) in log_dict[task_name]:
                    print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                    continue
                env_runner.make_env(episode_id)
                print(f"\n[robomme] env for task {task_name} episode {episode_id} setup finished")
                try:
                    success_flag = evaluator.eval_each_episode(env_runner, subgoal_predictor, video_save_dir)
                    if success_flag == "unknown":
                        log_dict[task_name][episode_id] = "error"
                    else:
                        log_dict[task_name][episode_id] = success_flag == "success"
                except Exception as e:
                    print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                    log_dict[task_name][episode_id] = "error"
                env_runner.close_env()
                with open(save_dir / "progress.json", "w") as f:
                    json.dump(log_dict, f, indent=2)
                if success_flag == "unknown":
                    print("API calling error, aborting...")
                    return
            del env_runner
            time.sleep(1)

        try:
            final_results = {
                "success_rate": {
                    t: sum(log_dict[t].values()) / len(log_dict[t].values())
                    for t in log_dict
                }
            }
            final_results["total_success_rate"] = (
                sum(final_results["success_rate"].values()) / len(final_results["success_rate"])
            )
            with open(save_dir / "log.json", "w") as f:
                json.dump(final_results, f, indent=2)
        except Exception as e:
            print(f"Error saving final results: {e}")
            time.sleep(1)


if __name__ == "__main__":
    import tyro
    tyro.cli(evaluate)
