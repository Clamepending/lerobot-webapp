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
PHASE_INFER_LOADING = "infer_loading"
PHASE_INFER_RUNNING = "infer_running"
# Remote inference 3-stage state machine:
PHASE_REMOTE_SPINUP = "remote_spinup"    # pod creating / installing / server starting
PHASE_REMOTE_READY = "remote_ready"      # pod live, server listening, waiting for Play
PHASE_REMOTE_PLAYING = "remote_playing"  # robot_client is running
PHASE_REMOTE_TEARDOWN = "remote_teardown"  # deleting pod


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


class InferenceRequest(BaseModel):
    policy_repo_id: str
    task: str = "move pingu to coaster"
    duration_s: float = 60.0
    fps: int = 30
    device: str = "cpu"  # "cpu" on Pi; no GPU available
    remote: bool = False  # if True, spin up a RunPod pod and run policy_server there
    actions_per_chunk: int = 50
    chunk_size_threshold: float = 0.5


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

        # Remote inference bookkeeping — set while a remote session is active
        self._remote_pod_id: Optional[str] = None
        self._remote_ssh: Optional[dict] = None          # {host, port}
        self._remote_server_url: Optional[str] = None    # localhost:PORT (SSH-tunneled)
        self._remote_tunnel_proc = None                  # the ssh -L tunnel subprocess
        self._remote_policy_repo_id: Optional[str] = None
        self._remote_client_proc = None                  # subprocess.Popen of robot_client
        self._remote_spinup_thread: Optional[threading.Thread] = None
        self._remote_lock = threading.Lock()

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

    def start_inference(self, req: InferenceRequest):
        if self.is_session_active():
            raise HTTPException(409, "a session is already running")
        missing_cams = [f"{n}={p}" for n, p in CAMERAS.items() if not Path(p).exists()]
        if missing_cams:
            raise HTTPException(400, "Cameras not present: " + ", ".join(missing_cams))
        self.events.update(stop_recording=False, exit_early=False)
        self._update(
            phase=PHASE_INFER_LOADING,
            repo_id=req.policy_repo_id,  # reuse field for policy
            task=req.task,
            num_episodes=0,
            episodes_recorded=0,
            current_episode=0,
            phase_started_at=0.0,
            phase_total_s=req.duration_s,
            fps=req.fps,
            total_frames=0,
            error=None,
            message=f"Loading policy {req.policy_repo_id}…",
            hub_url=f"https://huggingface.co/{req.policy_repo_id}",
        )
        self.worker = threading.Thread(
            target=self._run_inference, args=(req,), daemon=True, name="inference"
        )
        self.worker.start()

    def stop_inference(self):
        """E-stop: flip flags, kill any remote client subprocess, tear down pod."""
        self.events["stop_recording"] = True
        self.events["exit_early"] = True
        with self._remote_lock:
            proc = self._remote_client_proc
            pod_id = self._remote_pod_id
        if proc is not None:
            try:
                proc.terminate()
                try: proc.wait(timeout=3)
                except Exception: proc.kill()
            except Exception as exc:
                log.warning("failed terminating robot_client: %s", exc)
        if pod_id:
            threading.Thread(
                target=self._delete_pod, args=(pod_id,), daemon=True,
                name="pod_delete",
            ).start()

    @staticmethod
    def _delete_pod(pod_id: str):
        import subprocess
        try:
            subprocess.run(
                ["runpodctl", "pod", "delete", pod_id],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as exc:
            log.warning("pod delete failed (%s): %s", pod_id, exc)

    def _run_inference(self, req: InferenceRequest):
        """Local-only inference. Remote uses the 3-stage spinup/play/teardown flow."""
        self.stop_preview()
        try:
            self._inference_body(req)
        except Exception as exc:
            log.exception("inference failed")
            self._update(phase=PHASE_ERROR, error=str(exc), message=f"Error: {exc}")
        finally:
            self.start_preview()

    # -------- remote inference: 3-stage flow --------
    def remote_spinup(self, req: InferenceRequest):
        """Stage 1: create pod, install lerobot[async], start policy_server. Idempotent."""
        with self._remote_lock:
            if self._remote_server_url:
                return  # already spun up
            if self._remote_spinup_thread and self._remote_spinup_thread.is_alive():
                return  # in progress
        self.events.update(stop_recording=False, exit_early=False)
        self._remote_policy_repo_id = req.policy_repo_id
        self._update(
            phase=PHASE_REMOTE_SPINUP,
            repo_id=req.policy_repo_id,
            task=req.task,
            num_episodes=0, episodes_recorded=0, current_episode=0,
            phase_started_at=time.time(), phase_total_s=0,
            fps=req.fps, total_frames=0, error=None,
            message="Spinning up remote GPU…",
            hub_url=f"https://huggingface.co/{req.policy_repo_id}",
        )
        self.stop_preview()  # free cameras in case play comes next
        t = threading.Thread(
            target=self._do_spinup, args=(req,), daemon=True, name="remote_spinup"
        )
        self._remote_spinup_thread = t
        t.start()

    def remote_play(self, req: InferenceRequest):
        """Stage 2a: launch robot_client subprocess against the already-running server."""
        with self._remote_lock:
            if not self._remote_server_url:
                raise HTTPException(400, "remote not spun up yet — click Spin Up first")
            if self._remote_client_proc and self._remote_client_proc.poll() is None:
                return  # already playing
        import subprocess, shlex
        cams_arg = "{" + ", ".join(
            f"{n}: {{type: opencv, index_or_path: {p}, width: {CAM_WIDTH}, height: {CAM_HEIGHT}, fps: {CAM_FPS}}}"
            for n, p in CAMERAS.items()
        ) + "}"
        client_cmd = [
            "/home/jake/lerobot/.venv/bin/python", "-m", "lerobot.async_inference.robot_client",
            f"--server_address={self._remote_server_url}",
            "--robot.type=so101_follower",
            f"--robot.port={FOLLOWER_PORT}",
            f"--robot.id={FOLLOWER_ID}",
            f"--robot.cameras={cams_arg}",
            f"--task={req.task}",
            "--policy_type=act",
            f"--pretrained_name_or_path={self._remote_policy_repo_id}",
            "--policy_device=cuda",
            f"--actions_per_chunk={req.actions_per_chunk}",
            f"--chunk_size_threshold={req.chunk_size_threshold}",
        ]
        log.info("robot_client cmd: %s", " ".join(shlex.quote(c) for c in client_cmd))
        log_path = Path("/tmp/robot_client.log")
        log_path.write_text("")
        proc = subprocess.Popen(
            client_cmd,
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        with self._remote_lock:
            self._remote_client_proc = proc
        self._update(
            phase=PHASE_REMOTE_PLAYING,
            phase_started_at=time.time(), phase_total_s=req.duration_s,
            cameras=list(CAMERAS.keys()),
            message=f"Robot client running against {self._remote_server_url}",
        )
        threading.Thread(
            target=self._watch_client_proc, args=(proc, req, log_path),
            daemon=True, name="client_watcher",
        ).start()

    def _watch_client_proc(self, proc, req: InferenceRequest, log_path: Path):
        import subprocess
        start_t = time.time()
        while proc.poll() is None and not self.events["stop_recording"]:
            elapsed = time.time() - start_t
            if req.duration_s and elapsed >= req.duration_s:
                break
            with self.cond:
                self.state.total_frames = int(elapsed * req.fps)
                self.version += 1
                self.cond.notify_all()
            time.sleep(0.5)
        if proc.poll() is None:
            try:
                proc.terminate()
                try: proc.wait(timeout=5)
                except Exception: proc.kill()
            except Exception as exc:
                log.warning("client terminate failed: %s", exc)
        tail = log_path.read_text()[-400:] if log_path.exists() else ""
        with self._remote_lock:
            self._remote_client_proc = None
        # Go back to "ready" (pod still alive) unless an error occurred
        if proc.returncode not in (0, None, -15, 143):  # 15/-15 = SIGTERM (our stop)
            self._update(
                phase=PHASE_REMOTE_READY,
                message=f"robot_client exited {proc.returncode}. Tail: {tail[-200:]}",
            )
        else:
            self._update(
                phase=PHASE_REMOTE_READY,
                message=f"Stopped — pod still warm. Click Play to resume, Teardown to release.",
            )

    def remote_stop_play(self):
        """Stage 2b (E-STOP): kill robot_client immediately. Pod stays warm."""
        with self._remote_lock:
            proc = self._remote_client_proc
        self.events["stop_recording"] = True
        self.events["exit_early"] = True
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try: proc.wait(timeout=3)
                except Exception: proc.kill()
            except Exception as exc:
                log.warning("stop_play terminate failed: %s", exc)
        # _watch_client_proc will transition phase back to REMOTE_READY
        self.events["stop_recording"] = False  # reset so next play can proceed

    def remote_teardown(self):
        """Stage 3: kill the pod and reset all remote state."""
        with self._remote_lock:
            pod_id = self._remote_pod_id
            proc = self._remote_client_proc
            tunnel = self._remote_tunnel_proc
            self._remote_pod_id = None
            self._remote_server_url = None
            self._remote_ssh = None
            self._remote_client_proc = None
            self._remote_tunnel_proc = None
        self.events["stop_recording"] = True
        self.events["exit_early"] = True
        for p in (proc, tunnel):
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                    try: p.wait(timeout=3)
                    except Exception: p.kill()
                except Exception: pass
        self._update(phase=PHASE_REMOTE_TEARDOWN, message="Deleting pod…")
        if pod_id:
            self._delete_pod(pod_id)
        self.events["stop_recording"] = False
        self.events["exit_early"] = False
        self._update(phase=PHASE_DONE, message="Pod torn down.", cameras=[])
        # Resume local preview so cameras are live again
        self.start_preview()

    def _do_spinup(self, req: InferenceRequest):
        """Background thread body: create pod → install (detached, poll) → start server."""
        import json as _json
        import subprocess
        try:
            hf_tok = Path("/home/jake/.cache/huggingface/token").read_text().strip()
            env_json = _json.dumps({
                "HF_TOKEN": hf_tok,
                "HUGGING_FACE_HUB_TOKEN": hf_tok,
                "POLICY_REPO_ID": req.policy_repo_id,
            })

            # 1) create pod (try SECURE, then COMMUNITY)
            pod_id = None
            for cloud in ("SECURE", "COMMUNITY"):
                if self.events["stop_recording"]:
                    raise RuntimeError("cancelled")
                self._update(message=f"Creating 4090 pod ({cloud})…")
                cmd = [
                    "runpodctl", "pod", "create",
                    "--name", "lerobot-policy-server",
                    "--template-id", "runpod-torch-v240",
                    "--gpu-id", "NVIDIA GeForce RTX 4090",
                    "--container-disk-in-gb", "40",
                    "--volume-in-gb", "40",
                    "--cloud-type", cloud,
                    # Only SSH is needed publicly; we tunnel 8080 through ssh.
                    "--ports", "22/tcp",
                    "--env", env_json,
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                try:
                    info = _json.loads(r.stdout)
                except Exception:
                    info = {}
                if "id" in info:
                    pod_id = info["id"]; break
            if not pod_id:
                raise RuntimeError(f"pod create failed (both clouds): {r.stdout[:400]}")
            with self._remote_lock:
                self._remote_pod_id = pod_id
            self._update(message=f"Pod {pod_id} booting — waiting for SSH…")

            # 2) wait for SSH
            host, ssh_port = None, None
            deadline = time.time() + 240
            while time.time() < deadline and not self.events["stop_recording"]:
                r = subprocess.run(
                    ["runpodctl", "pod", "get", pod_id],
                    capture_output=True, text=True, timeout=30,
                )
                try: d = _json.loads(r.stdout)
                except Exception: d = {}
                ssh = d.get("ssh") or {}
                if (ssh.get("ip") or ssh.get("host")) and ssh.get("port"):
                    host = ssh.get("ip") or ssh.get("host")
                    ssh_port = ssh["port"]
                    break
                time.sleep(5)
            if not (host and ssh_port):
                raise RuntimeError("pod never became SSH-ready")
            with self._remote_lock:
                self._remote_ssh = {"host": host, "port": ssh_port}

            ssh_key = "/home/jake/.runpod/ssh/RunPod-Key-Go"
            ssh_base = [
                "ssh", "-i", ssh_key, "-p", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ServerAliveInterval=30",
                f"root@{host}",
            ]

            # 3) kick off install DETACHED on the pod, then poll for completion
            self._update(message="Installing lerobot on pod (detached)…")
            install_cmd = r"""
mkdir -p /workspace/logs
cat > /workspace/install.sh <<'EOS'
cd /workspace
export PATH=$HOME/.local/bin:$PATH
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
if [ ! -d lerobot ]; then git clone --depth 1 https://github.com/huggingface/lerobot lerobot; fi
cd lerobot
if [ ! -d .venv ]; then uv venv --python 3.12 .venv; fi
VENV=/workspace/lerobot/.venv
uv pip install --python $VENV/bin/python -e '.[async]' 2>&1
uv pip uninstall --python $VENV/bin/python torchcodec 2>/dev/null || true
HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | grep '^HF_TOKEN=' | cut -d= -f2-)
mkdir -p ~/.cache/huggingface
echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
echo INSTALL_OK > /workspace/logs/install.done
EOS
chmod +x /workspace/install.sh
rm -f /workspace/logs/install.done /workspace/logs/install.log
nohup bash /workspace/install.sh > /workspace/logs/install.log 2>&1 < /dev/null &
disown
echo INSTALL_STARTED
"""
            r = subprocess.run(ssh_base + [install_cmd], capture_output=True, text=True, timeout=120)
            if "INSTALL_STARTED" not in (r.stdout + r.stderr):
                raise RuntimeError(f"install kickoff failed: {(r.stdout + r.stderr)[-400:]}")

            # 4) poll for install completion — prefer last non-trace line + %progress if any
            deadline = time.time() + 900  # 15 min max
            last_msg = ""
            poll_cmd = (
                "test -f /workspace/logs/install.done && echo DONE || "
                # strip bash -x trace lines (start with '+') and pick the last useful line
                "grep -vE '^\\+' /workspace/logs/install.log 2>/dev/null | tail -1"
            )
            installed = False
            while time.time() < deadline and not self.events["stop_recording"]:
                r = subprocess.run(ssh_base + [poll_cmd], capture_output=True, text=True, timeout=30)
                if "DONE" in r.stdout:
                    installed = True
                    break
                msg = r.stdout.strip().replace("\n", " ")[:140]
                if msg and msg != last_msg:
                    self._update(message=f"Installing: {msg}")
                    last_msg = msg
                time.sleep(10)
            if not installed:
                raise RuntimeError("install timed out (>15 min)")
            if self.events["stop_recording"]:
                raise RuntimeError("cancelled during install")

            # 5) start policy_server on the pod — use absolute python path, capture PID,
            #    verify the process is actually alive after 5s (so we catch instant crashes)
            self._update(message="Starting policy_server on pod…")
            start_srv = r"""
VENV=/workspace/lerobot/.venv
rm -f /workspace/logs/server.log /workspace/logs/server.pid
cd /workspace/lerobot
nohup $VENV/bin/python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8080 \
  > /workspace/logs/server.log 2>&1 < /dev/null &
SRV_PID=$!
disown
echo $SRV_PID > /workspace/logs/server.pid
sleep 6
if kill -0 $SRV_PID 2>/dev/null; then
  echo SERVER_OK pid=$SRV_PID
else
  echo SERVER_DIED
  tail -40 /workspace/logs/server.log
fi
"""
            r = subprocess.run(ssh_base + [start_srv], capture_output=True, text=True, timeout=60)
            out = (r.stdout + r.stderr)
            if "SERVER_OK" not in out:
                tail = out[-600:]
                raise RuntimeError(f"policy_server didn't stay up. Tail: {tail}")

            # 6) open an SSH tunnel Pi:8080 → pod:8080 so robot_client can connect via localhost
            self._update(message="Opening SSH tunnel for policy_server…")
            tunnel_cmd = [
                "ssh", "-i", ssh_key, "-p", str(ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ServerAliveInterval=30",
                "-o", "ExitOnForwardFailure=yes",
                "-N",                           # no command
                "-L", "8080:localhost:8080",    # forward local 8080 → pod 8080
                f"root@{host}",
            ]
            tunnel = subprocess.Popen(
                tunnel_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
            )
            with self._remote_lock:
                self._remote_tunnel_proc = tunnel
            # Wait for tunnel to be usable: probe localhost:8080 with a TCP connect
            import socket
            ready = False
            for _ in range(30):  # ~15s
                if tunnel.poll() is not None:
                    err = (tunnel.stderr.read() or b"").decode(errors="replace")[-300:]
                    raise RuntimeError(f"ssh tunnel died: {err}")
                try:
                    s = socket.create_connection(("localhost", 8080), timeout=0.5)
                    s.close()
                    ready = True
                    break
                except OSError:
                    time.sleep(0.5)
            if not ready:
                raise RuntimeError("ssh tunnel opened but localhost:8080 never became connectable")

            server_url = "localhost:8080"
            with self._remote_lock:
                self._remote_server_url = server_url
            self._update(
                phase=PHASE_REMOTE_READY,
                message=f"Ready — policy_server on pod {pod_id} (tunneled). Click Play.",
            )
        except Exception as exc:
            log.exception("spinup failed")
            # Best-effort teardown of any partial pod + tunnel
            with self._remote_lock:
                pod_id = self._remote_pod_id
                tunnel = self._remote_tunnel_proc
                self._remote_pod_id = None
                self._remote_server_url = None
                self._remote_tunnel_proc = None
            if tunnel is not None and tunnel.poll() is None:
                try:
                    tunnel.terminate()
                    try: tunnel.wait(timeout=3)
                    except Exception: tunnel.kill()
                except Exception: pass
            if pod_id:
                self._delete_pod(pod_id)
            self._update(phase=PHASE_ERROR, error=str(exc), message=f"Spinup failed: {exc}")
            self.start_preview()

    def _inference_body(self, req: InferenceRequest):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.utils.control_utils import predict_action

        # Load policy config from the HF repo
        policy_cfg = PreTrainedConfig.from_pretrained(req.policy_repo_id)
        policy_cfg.device = req.device
        policy_cfg.pretrained_path = req.policy_repo_id

        # Build the robot (same follower + cameras we use for recording, no teleop)
        robot_config = SOFollowerRobotConfig(
            port=FOLLOWER_PORT,
            id=FOLLOWER_ID,
            cameras={
                name: OpenCVCameraConfig(
                    index_or_path=Path(path),
                    fps=CAM_FPS, width=CAM_WIDTH, height=CAM_HEIGHT,
                )
                for name, path in CAMERAS.items()
            },
        )
        robot = make_robot_from_config(robot_config)

        # Policy needs dataset meta (for feature shapes / stats) — we reuse the dataset
        # this policy was trained on. Find it from the training config.
        ds_repo = None
        try:
            from huggingface_hub import hf_hub_download
            import json as _json
            tc = hf_hub_download(req.policy_repo_id, "train_config.json", repo_type="model")
            ds_repo = _json.loads(Path(tc).read_text()).get("dataset", {}).get("repo_id")
        except Exception:
            pass
        if not ds_repo:
            raise RuntimeError("cannot determine training dataset from policy repo — provide manually")

        ds = LeRobotDataset(ds_repo)
        policy = make_policy(policy_cfg, ds_meta=ds.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=req.policy_repo_id,
            dataset_stats=ds.meta.stats,
            preprocessor_overrides={"device_processor": {"device": req.device}},
        )
        policy.eval()
        policy.reset()

        self._update(message=f"Connecting robot… ({len(robot_config.cameras)} cams)")
        robot.connect()
        self._update(
            cameras=list(robot.cameras.keys()),
            phase=PHASE_INFER_RUNNING,
            phase_started_at=time.time(),
            message=f"Running policy {req.policy_repo_id}",
        )

        device = torch.device(req.device)
        fps = req.fps
        start = time.perf_counter()
        timestamp = 0.0
        steps = 0
        last_ui_update = 0.0
        try:
            while timestamp < req.duration_s and not self.events["stop_recording"]:
                loop_start = time.perf_counter()
                obs = robot.get_observation()
                # Publish frames so MJPEG stays live
                for cam_name in robot.cameras:
                    frame = obs.get(cam_name)
                    if frame is None:
                        frame = obs.get(f"{OBS_STR}.images.{cam_name}")
                    if isinstance(frame, np.ndarray):
                        self._publish_frame(cam_name, frame)

                action_values = predict_action(
                    observation=obs,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=False,
                    task=req.task,
                    robot_type=robot.name,
                )
                robot_action = {
                    k: action_values[i].item()
                    for i, k in enumerate(robot.action_features)
                }
                robot.send_action(robot_action)

                steps += 1
                now = time.perf_counter()
                if now - last_ui_update > 0.25:
                    with self.cond:
                        self.state.total_frames = steps
                        self.version += 1
                        self.cond.notify_all()
                    last_ui_update = now

                dt = time.perf_counter() - loop_start
                precise_sleep(max(1 / fps - dt, 0.0))
                timestamp = time.perf_counter() - start
        finally:
            with contextlib.suppress(Exception):
                if robot.is_connected:
                    robot.disconnect()

        self._update(
            phase=PHASE_DONE,
            total_frames=steps,
            message=f"Inference done — {steps} steps in {timestamp:.1f}s",
        )

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


@app.get("/inference")
def inference_root():
    return FileResponse(STATIC_DIR / "inference.html")


@app.post("/api/inference/start")
def api_inference_start(req: InferenceRequest):
    controller.start_inference(req)
    return JSONResponse(controller.snapshot())


@app.post("/api/inference/stop")
def api_inference_stop():
    controller.stop_inference()
    return JSONResponse({"ok": True})


@app.post("/api/inference/remote/spinup")
def api_remote_spinup(req: InferenceRequest):
    controller.remote_spinup(req)
    return JSONResponse(controller.snapshot())


@app.post("/api/inference/remote/play")
def api_remote_play(req: InferenceRequest):
    controller.remote_play(req)
    return JSONResponse(controller.snapshot())


@app.post("/api/inference/remote/stop")
def api_remote_stop():
    controller.remote_stop_play()
    return JSONResponse({"ok": True})


@app.post("/api/inference/remote/teardown")
def api_remote_teardown():
    controller.remote_teardown()
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

def _viz_exclude_path(root: Path) -> Path:
    """Where the webapp stores its soft-delete manifest.

    Sits under meta/ alongside info.json; lerobot ignores unknown files here
    when loading the dataset, so it's a safe place for webapp-only state.
    """
    return root / "meta" / "webapp_excluded.json"

def _viz_load_excluded(root: Path) -> set[int]:
    p = _viz_exclude_path(root)
    if not p.exists():
        return set()
    try:
        import json
        d = json.loads(p.read_text())
        return set(int(i) for i in d.get("excluded", []))
    except Exception:
        log.warning("could not parse %s — treating as empty", p)
        return set()

def _viz_save_excluded(root: Path, excluded: set[int]):
    import json
    p = _viz_exclude_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"excluded": sorted(excluded)}, indent=2))

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
    root = _viz_resolve_repo(repo_id)
    excluded = _viz_load_excluded(root)
    kept_frames = sum(e["length"] for e in meta["episodes"] if e["episode_index"] not in excluded)
    return JSONResponse({
        "repo_id": repo_id,
        "fps": meta["info"].get("fps"),
        "robot_type": meta["info"].get("robot_type"),
        "total_episodes": meta["info"].get("total_episodes"),
        "total_frames": meta["info"].get("total_frames"),
        "kept_episodes": meta["info"].get("total_episodes", 0) - len(excluded),
        "kept_frames": kept_frames,
        "excluded": sorted(excluded),
        "video_keys": meta["video_keys"],
        "action_names": meta["info"]["features"].get("action", {}).get("names", []),
        "state_names": meta["info"]["features"].get("observation.state", {}).get("names", []),
        "episodes": [
            {
                "episode_index": e["episode_index"],
                "length": e["length"],
                "task": e["task"],
                "excluded": e["episode_index"] in excluded,
            }
            for e in meta["episodes"]
        ],
    })


class ToggleExcludeRequest(BaseModel):
    repo_id: str
    episode_index: int
    excluded: bool  # True = soft-delete (mark excluded). False = restore.


@app.post("/api/viz/toggle_exclude")
def viz_toggle_exclude(req: ToggleExcludeRequest):
    """Toggle the soft-delete flag on an episode. Instant + reversible.

    Nothing on disk moves — the episode is just flagged in
    meta/webapp_excluded.json so the UI can show it struck-through and
    the next Compact & Push will actually delete it.
    """
    root = _viz_resolve_repo(req.repo_id)
    meta = _viz_load_meta(req.repo_id)
    total = meta["info"].get("total_episodes", 0)
    if req.episode_index < 0 or req.episode_index >= total:
        raise HTTPException(400, f"invalid episode index {req.episode_index}")
    excluded = _viz_load_excluded(root)
    if req.excluded:
        excluded.add(req.episode_index)
    else:
        excluded.discard(req.episode_index)
    _viz_save_excluded(root, excluded)
    return JSONResponse({
        "ok": True,
        "excluded": sorted(excluded),
        "kept_episodes": total - len(excluded),
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


class PushRequest(BaseModel):
    repo_id: str


class CompactRequest(BaseModel):
    repo_id: str
    push: bool = True  # also mirror to HF after compacting


@app.post("/api/viz/compact")
def viz_compact(req: CompactRequest):
    """Apply all excluded-episode flags as a single atomic dataset rewrite,
    clear the manifest, and optionally push to HF. This is the ONLY action
    that causes irreversible changes to the local dataset or the remote.
    """
    import shutil
    from lerobot.datasets.dataset_tools import delete_episodes
    from huggingface_hub import HfApi

    root = _viz_resolve_repo(req.repo_id)
    excluded = sorted(_viz_load_excluded(root))

    ds = LeRobotDataset(req.repo_id, root=root)
    total_before = ds.meta.total_episodes

    result = {
        "ok": True,
        "excluded": excluded,
        "removed_count": 0,
        "kept_episodes": total_before,
        "kept_frames": ds.meta.total_frames,
        "pushed": False,
        "hub_url": None,
    }

    # 1) Apply exclusions (only if any)
    if excluded:
        if len(excluded) >= total_before:
            raise HTTPException(400, "cannot compact: all episodes are excluded")
        tmp_repo = f"{req.repo_id}__tmp_compact_{int(time.time())}"
        tmp_root = HF_LEROBOT_HOME / tmp_repo
        try:
            delete_episodes(
                dataset=ds,
                episode_indices=excluded,
                output_dir=tmp_root,
                repo_id=req.repo_id,
            )
        except Exception as exc:
            shutil.rmtree(tmp_root, ignore_errors=True)
            log.exception("compact: delete_episodes failed")
            raise HTTPException(500, f"compact failed: {exc}")

        backup = root.with_suffix(root.suffix + f".bak_{int(time.time())}")
        try:
            root.rename(backup)
            tmp_root.rename(root)
        except Exception as exc:
            if backup.exists() and not root.exists():
                backup.rename(root)
            shutil.rmtree(tmp_root, ignore_errors=True)
            raise HTTPException(500, f"compact swap failed: {exc}")
        shutil.rmtree(backup, ignore_errors=True)

        # manifest is now stale (indices re-numbered anyway); reset it
        _viz_save_excluded(root, set())
        _VIZ_META_CACHE.pop(req.repo_id, None)

        ds = LeRobotDataset(req.repo_id, root=root)
        result["removed_count"] = len(excluded)
        result["kept_episodes"] = ds.meta.total_episodes
        result["kept_frames"] = ds.meta.total_frames

    # 2) Push (mirror)
    if req.push:
        try:
            api = HfApi()
            api.create_repo(repo_id=req.repo_id, repo_type="dataset", exist_ok=True)
            # Don't push the webapp manifest; everything else mirrors.
            api.upload_folder(
                repo_id=req.repo_id,
                folder_path=str(root),
                repo_type="dataset",
                ignore_patterns=["images/", "meta/webapp_excluded.json"],
                delete_patterns=["*"],
                commit_message=(
                    f"Compact: removed {result['removed_count']} episode(s) → "
                    f"{result['kept_episodes']} eps / {result['kept_frames']} frames"
                ),
            )
            result["pushed"] = True
            result["hub_url"] = f"https://huggingface.co/datasets/{req.repo_id}"
        except Exception as exc:
            log.exception("compact: push failed")
            # Partial success: local compact happened, remote push did not.
            raise HTTPException(500, f"compact succeeded locally but push failed: {exc}")

    return JSONResponse(result)


@app.post("/api/viz/push_to_hub")
def viz_push_to_hub(req: PushRequest):
    """Push the local dataset up to HuggingFace as a mirror.

    Uses delete_patterns=["*"] so files that exist on the remote but no
    longer exist locally (e.g. from episode deletions that shrank the
    file count) are removed — lerobot's own push_to_hub only adds/overwrites,
    so deletions weren't propagating.
    """
    from huggingface_hub import HfApi
    root = _viz_resolve_repo(req.repo_id)
    try:
        ds = LeRobotDataset(req.repo_id, root=root)
        api = HfApi()
        api.create_repo(repo_id=req.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=req.repo_id,
            folder_path=str(root),
            repo_type="dataset",
            ignore_patterns=["images/"],
            delete_patterns=["*"],  # mirror: wipe remote files not present locally
            commit_message=f"Sync from lerobot-webapp ({ds.meta.total_episodes} eps, "
                           f"{ds.meta.total_frames} frames)",
        )
    except Exception as exc:
        log.exception("push_to_hub failed")
        raise HTTPException(500, f"push failed: {exc}")
    return JSONResponse({
        "ok": True,
        "hub_url": f"https://huggingface.co/datasets/{req.repo_id}",
        "episodes": ds.meta.total_episodes,
        "frames": ds.meta.total_frames,
    })


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
