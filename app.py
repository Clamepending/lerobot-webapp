"""Web UI for recording LeRobot datasets on the SO-ARM 101.

Wraps the lerobot recording machinery in a FastAPI server so the user sees:
  - live camera stream
  - a big visible phase indicator (IDLE / RECORDING / RESET / SAVING / DONE)
  - episode counter + countdown
  - keyboard shortcuts that actually drive the recording loop

The webapp owns the hardware (leader, follower, cameras) for the duration of
the session. Don't run `lerobot-record` in parallel — they'd fight over the
USB ports.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

init_logging()
log = logging.getLogger("lerobot-webapp")

FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "/dev/ttyACM1")
LEADER_PORT = os.environ.get("LEADER_PORT", "/dev/ttyACM0")
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "my_follower")
LEADER_ID = os.environ.get("LEADER_ID", "my_leader")
CAM_WIDTH = int(os.environ.get("CAM_WIDTH", "640"))
CAM_HEIGHT = int(os.environ.get("CAM_HEIGHT", "480"))
CAM_FPS = int(os.environ.get("CAM_FPS", "30"))

# Cameras to attach to the follower. Override via CAMERAS env var, format:
#   CAMERAS="wrist=/dev/video0,top=/dev/video2"
def _parse_cameras() -> dict[str, str]:
    raw = os.environ.get("CAMERAS", "wrist=/dev/video0,top=/dev/video2")
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, path = part.partition("=")
        if not path:
            continue
        if Path(path).exists():
            out[name.strip()] = path.strip()
        else:
            log.warning("camera %s=%s skipped (device not present)", name, path)
    return out

CAMERAS = _parse_cameras()

PHASE_IDLE = "idle"
PHASE_STARTING = "starting"
PHASE_READY = "ready"
PHASE_RECORDING = "recording"
PHASE_RESET = "reset"
PHASE_SAVING = "saving"
PHASE_UPLOADING = "uploading"
PHASE_DONE = "done"
PHASE_ERROR = "error"


class SessionRequest(BaseModel):
    repo_id: str
    task: str
    num_episodes: int = 30
    episode_time_s: float = 30
    reset_time_s: float = 10
    resume: bool = False
    push_to_hub: bool = True
    fps: int = 30


class CommandRequest(BaseModel):
    command: str  # advance | re_record | abort | start


@dataclass
class State:
    phase: str = PHASE_IDLE
    repo_id: str = ""
    task: str = ""
    num_episodes: int = 0
    episodes_recorded: int = 0
    current_episode: int = 0  # 1-based display index of the episode currently being handled
    phase_started_at: float = 0.0
    phase_total_s: float = 0.0
    fps: int = 30
    total_frames: int = 0
    error: Optional[str] = None
    cameras: list[str] = field(default_factory=list)
    push_to_hub: bool = True
    resume: bool = False
    message: str = ""
    hub_url: str = ""


def phase_elapsed(st: State) -> float:
    if st.phase_started_at == 0.0:
        return 0.0
    return max(0.0, time.time() - st.phase_started_at)


class Controller:
    """Owns the hardware and the recording worker thread."""

    def __init__(self):
        self.state = State()
        self.lock = threading.RLock()
        self.version = 0
        self.cond = threading.Condition(self.lock)

        # mutable events dict shared with the record loop
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "waiting_for_start": False,
            "start_episode": False,
        }

        self.latest_jpeg: dict[str, bytes] = {}
        self.jpeg_version: dict[str, int] = {}
        self.jpeg_cond = threading.Condition()

        self.worker: Optional[threading.Thread] = None

        # Preview manager: keeps cameras live outside of a recording session so the
        # operator can aim them in real time. We hand the cameras over to lerobot
        # during a session (v4l2 doesn't like double-open), then reclaim them.
        self._preview_stop = threading.Event()
        self._preview_stop.set()  # starts in stopped state; _ensure_preview flips it
        self._preview_threads: dict[str, threading.Thread] = {}
        self._preview_lock = threading.Lock()

    # -------- state --------
    def snapshot(self) -> dict:
        with self.lock:
            d = {k: getattr(self.state, k) for k in self.state.__dataclass_fields__}
            d["phase_elapsed_s"] = phase_elapsed(self.state)
            d["version"] = self.version
            d["session_active"] = self.is_session_active()
            return d

    def _update(self, **kwargs):
        with self.cond:
            for k, v in kwargs.items():
                setattr(self.state, k, v)
            self.version += 1
            self.cond.notify_all()

    def wait_for_change(self, last_version: int, timeout: float = 2.0) -> int:
        with self.cond:
            if self.version == last_version:
                self.cond.wait(timeout=timeout)
            return self.version

    # -------- frame streaming --------
    def _publish_frame(self, cam: str, rgb: np.ndarray):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        jpeg = buf.tobytes()
        with self.jpeg_cond:
            self.latest_jpeg[cam] = jpeg
            self.jpeg_version[cam] = self.jpeg_version.get(cam, 0) + 1
            self.jpeg_cond.notify_all()

    def wait_jpeg(self, cam: str, last: int, timeout: float = 0.5) -> tuple[Optional[bytes], int]:
        with self.jpeg_cond:
            if self.jpeg_version.get(cam, 0) == last:
                self.jpeg_cond.wait(timeout=timeout)
            return self.latest_jpeg.get(cam), self.jpeg_version.get(cam, 0)

    # -------- preview (cameras live while no session is running) --------
    def _preview_loop(self, name: str, path: str):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        if not cap.isOpened():
            log.warning("preview: failed to open %s (%s)", name, path)
            return
        log.info("preview: %s (%s) opened", name, path)
        try:
            while not self._preview_stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                # _publish_frame expects RGB; our cv2 read is BGR, so swap before publish
                # (publish will re-swap back to BGR for JPEG encode — net: correct output)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self._publish_frame(name, frame_rgb)
        finally:
            cap.release()
            log.info("preview: %s released", name)

    def start_preview(self):
        """Open cv2.VideoCapture on every configured camera and stream to /stream/<name>.

        Re-reads CAMERAS mapping and skips devices that aren't currently present so
        a USB disconnect doesn't leave us trying to open a vanished /dev/videoN.
        """
        with self._preview_lock:
            if not self._preview_stop.is_set():
                return  # already running
            self._preview_stop.clear()
            # Re-scan: a camera may have dropped off since last time.
            available = {
                name: path for name, path in CAMERAS.items() if Path(path).exists()
            }
            missing = [n for n in CAMERAS if n not in available]
            if missing:
                log.warning("preview: cameras not present, skipping: %s", missing)
            self._preview_threads = {}
            for name, path in available.items():
                t = threading.Thread(
                    target=self._preview_loop, args=(name, path), daemon=True,
                    name=f"preview_{name}",
                )
                t.start()
                self._preview_threads[name] = t
            # Only expose cameras that actually exist so the frontend doesn't subscribe
            # to a dead /stream/<name>.
            self._update(cameras=list(available.keys()))

    def stop_preview(self, timeout: float = 3.0):
        """Stop preview threads and release camera handles so lerobot can open them."""
        with self._preview_lock:
            if self._preview_stop.is_set():
                return
            self._preview_stop.set()
            threads = list(self._preview_threads.values())
            self._preview_threads = {}
        for t in threads:
            t.join(timeout=timeout)

    # -------- session control --------
    def is_session_active(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def start_session(self, req: SessionRequest):
        if self.is_session_active():
            raise HTTPException(409, "a session is already running")
        # Fail fast if any configured camera is missing (e.g. USB cable came loose).
        missing_cams = [f"{n}={p}" for n, p in CAMERAS.items() if not Path(p).exists()]
        if missing_cams:
            raise HTTPException(
                400,
                "Cameras not present: " + ", ".join(missing_cams) +
                ". Reconnect the cable(s) or remove them from the CAMERAS env var.",
            )
        dataset_root = HF_LEROBOT_HOME / req.repo_id
        if dataset_root.exists() and not req.resume:
            raise HTTPException(
                400,
                f"dataset already exists at {dataset_root}. tick 'Resume' to add episodes to it, "
                "or delete the folder first to start fresh.",
            )
        if req.resume and not dataset_root.exists():
            raise HTTPException(
                400,
                f"no dataset found at {dataset_root} to resume. untick 'Resume' to create it fresh.",
            )
        self.events.update(
            exit_early=False,
            rerecord_episode=False,
            stop_recording=False,
            waiting_for_start=False,
            start_episode=False,
        )
        self._update(
            phase=PHASE_STARTING,
            repo_id=req.repo_id,
            task=req.task,
            num_episodes=req.num_episodes,
            episodes_recorded=0,
            current_episode=0,
            phase_started_at=0.0,
            phase_total_s=0.0,
            fps=req.fps,
            total_frames=0,
            error=None,
            push_to_hub=req.push_to_hub,
            resume=req.resume,
            message="Connecting to hardware…",
            hub_url="",
        )
        self.worker = threading.Thread(target=self._run_session, args=(req,), daemon=True)
        self.worker.start()

    def command(self, cmd: str):
        if cmd == "start":
            # Release the "waiting_for_start" gate so the next episode begins.
            self.events["start_episode"] = True
        elif cmd == "advance":
            self.events["exit_early"] = True
            # If we were gated waiting for user to start, also kick us out of that gate.
            self.events["start_episode"] = True
        elif cmd == "re_record":
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
        elif cmd == "abort":
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
            self.events["start_episode"] = True  # unblock the gate if any
        else:
            raise HTTPException(400, f"unknown command: {cmd}")

    # -------- worker --------
    def _run_session(self, req: SessionRequest):
        # Hand cameras over to lerobot for the duration of the session.
        self.stop_preview()
        try:
            self._session_body(req)
        except Exception as exc:
            log.exception("session failed")
            self._update(phase=PHASE_ERROR, error=str(exc), message=f"Error: {exc}")
        finally:
            # Reclaim cameras so the viewport stays live between sessions.
            self.start_preview()

    def _session_body(self, req: SessionRequest):
        robot_config = SOFollowerRobotConfig(
            port=FOLLOWER_PORT,
            id=FOLLOWER_ID,
            cameras={
                name: OpenCVCameraConfig(
                    index_or_path=Path(path),
                    fps=CAM_FPS,
                    width=CAM_WIDTH,
                    height=CAM_HEIGHT,
                )
                for name, path in CAMERAS.items()
            },
        )
        teleop_config = SO101LeaderConfig(
            port=LEADER_PORT,
            id=LEADER_ID,
        )

        robot = make_robot_from_config(robot_config)
        teleop = make_teleoperator_from_config(teleop_config)

        teleop_ap, robot_ap, robot_op = make_default_processors()

        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_ap,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_op,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        )

        if req.resume:
            dataset = LeRobotDataset(req.repo_id, batch_encoding_size=1)
            dataset.start_image_writer(num_processes=0, num_threads=4 * len(robot.cameras))
            sanity_check_dataset_robot_compatibility(dataset, robot, req.fps, dataset_features)
        else:
            sanity_check_dataset_name(req.repo_id, None)
            dataset = LeRobotDataset.create(
                req.repo_id,
                req.fps,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4 * len(robot.cameras),
                batch_encoding_size=1,
            )

        self._update(message="Connecting robot + teleop…")
        robot.connect()
        teleop.connect()
        self._update(
            cameras=list(robot.cameras.keys()),
            episodes_recorded=dataset.num_episodes,
            total_frames=dataset.num_frames,
        )

        try:
            with VideoEncodingManager(dataset):
                self._record_episodes(req, robot, teleop, dataset, teleop_ap, robot_ap, robot_op)
        finally:
            self._update(phase=PHASE_SAVING, message="Finalizing dataset…")
            with contextlib.suppress(Exception):
                dataset.finalize()
            with contextlib.suppress(Exception):
                if robot.is_connected:
                    robot.disconnect()
            with contextlib.suppress(Exception):
                if teleop.is_connected:
                    teleop.disconnect()

        if req.push_to_hub and not self.events["stop_recording"]:
            self._update(phase=PHASE_UPLOADING, message="Pushing to Hugging Face…")
            try:
                dataset.push_to_hub()
                hub_url = f"https://huggingface.co/datasets/{req.repo_id}"
                self._update(hub_url=hub_url, message=f"Pushed → {hub_url}")
            except Exception as exc:
                log.exception("push_to_hub failed")
                self._update(message=f"Push failed: {exc}")

        self._update(
            phase=PHASE_DONE,
            message=f"Done — {dataset.num_episodes} episodes, {dataset.num_frames} frames",
            episodes_recorded=dataset.num_episodes,
            total_frames=dataset.num_frames,
        )

    def _record_episodes(self, req, robot, teleop, dataset, teleop_ap, robot_ap, robot_op):
        recorded_this_session = 0
        target_total = req.num_episodes  # meaning: total episodes in the dataset

        while dataset.num_episodes < target_total and not self.events["stop_recording"]:
            # Ready gate — wait for user to press Start / Space before each episode.
            # Pass robot so the ready gate keeps the MJPEG stream live.
            self._wait_for_ready(dataset.num_episodes + 1, target_total, robot=robot, robot_op=robot_op)
            if self.events["stop_recording"]:
                break

            # RECORDING phase
            self._update(
                phase=PHASE_RECORDING,
                current_episode=dataset.num_episodes + 1,
                phase_started_at=time.time(),
                phase_total_s=req.episode_time_s,
                message=f"Recording episode {dataset.num_episodes + 1}",
            )
            self._run_loop(
                robot, teleop, dataset, teleop_ap, robot_ap, robot_op,
                control_time_s=req.episode_time_s,
                single_task=req.task,
                write_to_dataset=True,
            )

            if self.events["rerecord_episode"]:
                self._update(phase=PHASE_SAVING, message="Discarding episode buffer…")
                dataset.clear_episode_buffer()
                self.events["rerecord_episode"] = False
                self.events["exit_early"] = False
                continue

            if self.events["stop_recording"]:
                break

            # Optional RESET phase (skip on the very last episode)
            is_last = dataset.num_episodes + 1 >= target_total
            if not is_last and req.reset_time_s > 0:
                self._update(
                    phase=PHASE_RESET,
                    phase_started_at=time.time(),
                    phase_total_s=req.reset_time_s,
                    message="Reset the environment",
                )
                self._run_loop(
                    robot, teleop, dataset, teleop_ap, robot_ap, robot_op,
                    control_time_s=req.reset_time_s,
                    single_task=req.task,
                    write_to_dataset=False,
                )

            if self.events["stop_recording"]:
                break

            # Save the episode we just recorded
            self._update(phase=PHASE_SAVING, message="Saving episode…")
            dataset.save_episode()
            recorded_this_session += 1
            self._update(
                episodes_recorded=dataset.num_episodes,
                total_frames=dataset.num_frames,
            )

    def _wait_for_ready(self, episode_num: int, target_total: int, robot=None, robot_op=None):
        self.events["start_episode"] = False
        self._update(
            phase=PHASE_READY,
            current_episode=episode_num,
            phase_started_at=0.0,
            phase_total_s=0.0,
            message=f"Ready for episode {episode_num} of {target_total} — press Space / Start",
        )
        # While gated, keep pulling frames at ~15 fps so the MJPEG stream stays live
        # (otherwise the viewport freezes on the last recorded frame).
        frame_interval = 1 / 15
        next_pull = time.perf_counter()
        while not self.events["start_episode"] and not self.events["stop_recording"]:
            if robot is not None and time.perf_counter() >= next_pull:
                try:
                    obs = robot.get_observation()
                    obs_proc = robot_op(obs) if robot_op else obs
                    for cam_name in robot.cameras:
                        frame = obs_proc.get(cam_name)
                        if frame is None:
                            frame = obs_proc.get(f"{OBS_STR}.images.{cam_name}")
                        if isinstance(frame, np.ndarray):
                            self._publish_frame(cam_name, frame)
                except TimeoutError:
                    pass  # transient camera stall; try again next tick
                except Exception as exc:
                    log.warning("ready-gate frame pull failed: %s", exc)
                next_pull = time.perf_counter() + frame_interval
            time.sleep(0.02)
        self.events["start_episode"] = False
        self.events["exit_early"] = False

    def _run_loop(
        self, robot, teleop, dataset, teleop_ap, robot_ap, robot_op,
        *, control_time_s: float, single_task: str, write_to_dataset: bool,
    ):
        """Tight loop: read obs, publish frame, teleop action, send action, (optionally) record."""
        fps = dataset.fps
        start = time.perf_counter()
        timestamp = 0.0
        last_ui_update = 0.0
        camera_miss_streak = 0
        while timestamp < control_time_s:
            loop_start = time.perf_counter()
            if self.events["exit_early"]:
                self.events["exit_early"] = False
                break

            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                # Pi 5 USB cameras sometimes stall for 300-600ms. Don't crash the session:
                # skip this frame and keep going. If misses stack up, surface it.
                camera_miss_streak += 1
                log.warning("camera stall #%d: %s", camera_miss_streak, e)
                if camera_miss_streak >= 30:  # ~1s at 30fps
                    raise
                precise_sleep(1 / fps)
                timestamp = time.perf_counter() - start
                continue
            except RuntimeError as e:
                # Lerobot's camera read thread dies after ~11 consecutive hw failures
                # (classic USB disconnect: errno=19 ENODEV). We can't recover without
                # reconnecting the device, so surface a clear message and stop the session.
                msg = str(e)
                if "read thread is not running" in msg or "read failed" in msg:
                    raise RuntimeError(
                        "A camera stopped responding mid-session (likely USB disconnect). "
                        "Reconnect the camera cable and start a new session. "
                        f"Original error: {msg}"
                    ) from e
                raise
            camera_miss_streak = 0
            obs_processed = robot_op(obs)

            # Publish camera frames to MJPEG subscribers (don't do it every frame if fps is high)
            for cam_name in robot.cameras:
                frame = obs_processed.get(cam_name)
                if frame is None:
                    # Some processors flatten keys; try the prefixed form
                    frame = obs_processed.get(f"{OBS_STR}.images.{cam_name}")
                if isinstance(frame, np.ndarray):
                    self._publish_frame(cam_name, frame)

            if write_to_dataset:
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

            act = teleop.get_action()
            act_processed = teleop_ap((act, obs))
            robot_action = robot_ap((act_processed, obs))
            robot.send_action(robot_action)

            if write_to_dataset:
                action_frame = build_dataset_frame(dataset.features, act_processed, prefix=ACTION)
                frame_to_add = {**observation_frame, **action_frame, "task": single_task}
                dataset.add_frame(frame_to_add)

            # Update phase elapsed at ~4Hz so the UI clock ticks smoothly without spamming the condvar.
            now = time.perf_counter()
            if now - last_ui_update > 0.25:
                with self.cond:
                    self.version += 1
                    self.cond.notify_all()
                last_ui_update = now

            dt = time.perf_counter() - loop_start
            precise_sleep(max(1 / fps - dt, 0.0))
            timestamp = time.perf_counter() - start


controller = Controller()
# Open cameras immediately so the UI is live before anyone starts a session.
controller.start_preview()

app = FastAPI(title="LeRobot Recording UI")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/state")
def get_state():
    return JSONResponse(controller.snapshot())


@app.get("/api/state/stream")
def state_stream():
    """Server-Sent Events stream of state snapshots."""

    def gen():
        last = -1
        # Send an initial snapshot immediately.
        snap = controller.snapshot()
        last = snap["version"]
        yield f"data: {_json(snap)}\n\n"
        while True:
            last = controller.wait_for_change(last, timeout=10.0)
            snap = controller.snapshot()
            yield f"data: {_json(snap)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/session/start")
def api_start(req: SessionRequest):
    controller.start_session(req)
    return JSONResponse(controller.snapshot())


@app.post("/api/session/command")
def api_command(req: CommandRequest):
    controller.command(req.command)
    return JSONResponse({"ok": True})


@app.get("/stream/{cam}")
def stream(cam: str):
    boundary = b"--frame"

    def gen():
        last = -1
        deadline = time.time() + 1.5
        while controller.latest_jpeg.get(cam) is None and time.time() < deadline:
            time.sleep(0.1)
        while True:
            jpeg, last = controller.wait_jpeg(cam, last, timeout=2.0)
            if jpeg is None:
                # no frame available yet; send a tiny placeholder so browsers don't 0-byte
                time.sleep(0.1)
                continue
            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            yield jpeg
            yield b"\r\n"

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )


# ==================== Dataset visualizer ====================
# Browses any LeRobotDataset sitting in HF_LEROBOT_HOME. Reads the per-episode
# meta parquet for video file pointers + timestamp ranges, reads the data
# parquet for action/observation.state timeseries, serves chunk MP4s with
# HTTP range support so the browser can seek directly to each episode.

_VIZ_META_CACHE: dict[str, dict] = {}

def _viz_resolve_repo(repo_id: str) -> Path:
    root = HF_LEROBOT_HOME / repo_id
    if not root.exists():
        raise HTTPException(404, f"dataset not found locally at {root}. pull it first.")
    return root

def _viz_load_meta(repo_id: str) -> dict:
    """Load + cache per-episode metadata (video file ptrs, frame ranges, task)."""
    if repo_id in _VIZ_META_CACHE:
        return _VIZ_META_CACHE[repo_id]
    import pandas as pd
    root = _viz_resolve_repo(repo_id)
    info = _json_load(root / "meta" / "info.json")
    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise HTTPException(500, "no episode metadata found")
    df = pd.concat([pd.read_parquet(f) for f in ep_files]).sort_values("episode_index")
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    episodes = []
    for _, row in df.iterrows():
        vids = {}
        for vk in video_keys:
            vids[vk] = {
                "chunk_index": int(row[f"videos/{vk}/chunk_index"]),
                "file_index": int(row[f"videos/{vk}/file_index"]),
                "from_timestamp": float(row[f"videos/{vk}/from_timestamp"]),
                "to_timestamp": float(row[f"videos/{vk}/to_timestamp"]),
            }
        task = row["tasks"]
        if hasattr(task, "tolist"):
            task = task.tolist()
        if isinstance(task, list):
            task = ", ".join(task)
        episodes.append({
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "task": str(task),
            "data_chunk": int(row["data/chunk_index"]),
            "data_file": int(row["data/file_index"]),
            "dataset_from_index": int(row["dataset_from_index"]),
            "dataset_to_index": int(row["dataset_to_index"]),
            "videos": vids,
        })
    meta = {"info": info, "episodes": episodes, "video_keys": video_keys}
    _VIZ_META_CACHE[repo_id] = meta
    return meta

def _json_load(path: Path) -> dict:
    import json
    return json.loads(Path(path).read_text())

@app.get("/viz")
def viz_root():
    return FileResponse(STATIC_DIR / "viz.html")

@app.get("/api/viz/info")
def viz_info(repo_id: str = "clamepending/pingu_to_coaster_420"):
    meta = _viz_load_meta(repo_id)
    return JSONResponse({
        "repo_id": repo_id,
        "fps": meta["info"].get("fps"),
        "robot_type": meta["info"].get("robot_type"),
        "total_episodes": meta["info"].get("total_episodes"),
        "total_frames": meta["info"].get("total_frames"),
        "video_keys": meta["video_keys"],
        "action_names": meta["info"]["features"].get("action", {}).get("names", []),
        "state_names": meta["info"]["features"].get("observation.state", {}).get("names", []),
        "episodes": [
            {"episode_index": e["episode_index"], "length": e["length"], "task": e["task"]}
            for e in meta["episodes"]
        ],
    })

@app.get("/api/viz/episode/{episode_index}")
def viz_episode(episode_index: int, repo_id: str = "clamepending/pingu_to_coaster_420"):
    import pandas as pd
    meta = _viz_load_meta(repo_id)
    ep = next((e for e in meta["episodes"] if e["episode_index"] == episode_index), None)
    if ep is None:
        raise HTTPException(404, f"episode {episode_index} not found")
    root = _viz_resolve_repo(repo_id)
    parquet = root / "data" / f"chunk-{ep['data_chunk']:03d}" / f"file-{ep['data_file']:03d}.parquet"
    df = pd.read_parquet(parquet)
    df = df[df["episode_index"] == episode_index].sort_values("frame_index")
    def stack(col: str) -> list[list[float]]:
        return [list(map(float, v)) for v in df[col].tolist()]
    return JSONResponse({
        "episode_index": episode_index,
        "length": len(df),
        "task": ep["task"],
        "timestamps": df["timestamp"].astype(float).tolist(),
        "action": stack("action"),
        "observation_state": stack("observation.state"),
        "videos": ep["videos"],
    })

class DeleteEpisodesRequest(BaseModel):
    repo_id: str
    episode_indices: list[int]


@app.post("/api/viz/delete_episodes")
def viz_delete_episodes(req: DeleteEpisodesRequest):
    """Delete episodes from a dataset in place (atomic swap with a temp copy).

    Uses lerobot's delete_episodes which builds a reindexed copy, then we swap
    the new copy into the original repo path so all subsequent reads see the
    updated dataset under the same repo_id.
    """
    import shutil
    from lerobot.datasets.dataset_tools import delete_episodes
    root = _viz_resolve_repo(req.repo_id)
    if not req.episode_indices:
        raise HTTPException(400, "no episode_indices provided")
    ds = LeRobotDataset(req.repo_id, root=root)
    total = ds.meta.total_episodes
    invalid = [i for i in req.episode_indices if i < 0 or i >= total]
    if invalid:
        raise HTTPException(400, f"invalid episode indices: {invalid}")
    if len(req.episode_indices) >= total:
        raise HTTPException(400, "cannot delete all episodes; dataset would be empty")

    tmp_repo = f"{req.repo_id}__tmp_del_{int(time.time())}"
    tmp_root = HF_LEROBOT_HOME / tmp_repo
    try:
        delete_episodes(
            dataset=ds,
            episode_indices=sorted(set(req.episode_indices)),
            output_dir=tmp_root,
            repo_id=req.repo_id,  # keep original repo_id in the metadata
        )
    except Exception as exc:
        shutil.rmtree(tmp_root, ignore_errors=True)
        log.exception("delete_episodes failed")
        raise HTTPException(500, f"delete failed: {exc}")

    # Atomic swap: move original out of the way, move new in, then delete backup.
    backup = root.with_suffix(root.suffix + f".bak_{int(time.time())}")
    try:
        root.rename(backup)
        tmp_root.rename(root)
    except Exception as exc:
        # Try to roll back
        if backup.exists() and not root.exists():
            backup.rename(root)
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise HTTPException(500, f"swap failed: {exc}")
    shutil.rmtree(backup, ignore_errors=True)

    # Invalidate cached meta so the next /api/viz/info re-reads from disk.
    _VIZ_META_CACHE.pop(req.repo_id, None)

    # Re-read new totals for response
    new_ds = LeRobotDataset(req.repo_id, root=root)
    return JSONResponse({
        "ok": True,
        "deleted": sorted(set(req.episode_indices)),
        "remaining_episodes": new_ds.meta.total_episodes,
        "remaining_frames": new_ds.meta.total_frames,
    })


@app.get("/api/viz/video/{camera}/{episode_index}")
def viz_video(camera: str, episode_index: int, repo_id: str = "clamepending/pingu_to_coaster_420"):
    """Serve the chunk MP4 that contains this episode, with HTTP range support
    so the browser can seek directly to from_timestamp."""
    meta = _viz_load_meta(repo_id)
    ep = next((e for e in meta["episodes"] if e["episode_index"] == episode_index), None)
    if ep is None:
        raise HTTPException(404, f"episode {episode_index} not found")
    vk = camera if camera.startswith("observation.images.") else f"observation.images.{camera}"
    if vk not in ep["videos"]:
        raise HTTPException(404, f"camera {camera} not in dataset (have: {list(ep['videos'])})")
    v = ep["videos"][vk]
    root = _viz_resolve_repo(repo_id)
    mp4 = root / "videos" / vk / f"chunk-{v['chunk_index']:03d}" / f"file-{v['file_index']:03d}.mp4"
    if not mp4.exists():
        raise HTTPException(404, f"video file missing: {mp4}")
    # Starlette's FileResponse supports HTTP Range by default.
    return FileResponse(
        mp4, media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            # expose segment bounds for the frontend
            "X-Episode-From-Ts": str(v["from_timestamp"]),
            "X-Episode-To-Ts": str(v["to_timestamp"]),
            "Access-Control-Expose-Headers": "X-Episode-From-Ts, X-Episode-To-Ts",
        },
    )


def _json(d: dict) -> str:
    import json
    return json.dumps(d, default=str)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
