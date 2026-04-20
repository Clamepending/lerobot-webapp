# lerobot-webapp

A FastAPI + vanilla JS web UI for recording and visualizing
[LeRobot](https://github.com/huggingface/lerobot) datasets on an SO-ARM 101
(or any robot following the same interface) running on a Raspberry Pi 5.

Replaces `lerobot-record` with a browser-driven UX tuned for solo data
collection sessions: live dual-camera preview, big phase indicator, countdown,
episode timeline, keyboard shortcuts.

## Features

### Recorder (`/`)

- **Live camera preview** — cameras are open the moment the webapp boots, so
  you can aim them before starting a session.
- **Phase-driven UX** — `IDLE / STARTING / READY / RECORDING / RESET / SAVING /
  DONE`, with color-coded pill, 48px tabular countdown, and a row of episode
  dots that fill as you go.
- **Start gate per episode** — press Space (or the button) to begin each
  recording; no surprise timer-driven starts.
- **Keyboard shortcuts**
  - `Space` — start episode / end early
  - `R` — re-record current episode (discard buffer)
  - `Q` — abort session (preserves recorded episodes)
- **Resume-aware** — ticks a checkbox in the dialog to continue adding
  episodes to an existing dataset.
- **Dual-cam support** — wrist + top (or any number) rendered side-by-side,
  labeled, during both preview and recording.
- **Resilient to USB hiccups** — catches transient camera stalls (300–600 ms
  gaps are common on Pi 5) without crashing the session; surfaces a clear
  error if a camera physically disconnects.

### Visualizer (`/viz`)

Inspect any `LeRobotDataset` sitting in `~/.cache/huggingface/lerobot/…`:

- Side-by-side video playback of every camera in the dataset.
- Synced timeseries charts for `action` and `observation.state` (6-DoF),
  each channel in its own color with a live cursor that follows the video.
- Scrubber, play/pause, 0.5×/1×/2×/4× speeds, loop toggle.
- `Space` and `←/→` keyboard shortcuts for frame-level navigation.
- Defaults to the `clamepending/pingu_to_coaster_420` dataset; override with
  `?repo_id=user/name`.

## Run

```bash
# From inside a lerobot venv (activate or point directly)
python app.py
# then open http://<pi-ip>:8000/
```

Env overrides:

```bash
FOLLOWER_PORT=/dev/ttyACM1
LEADER_PORT=/dev/ttyACM0
FOLLOWER_ID=my_follower
LEADER_ID=my_leader
CAMERAS="wrist=/dev/video0,top=/dev/video2"
CAM_WIDTH=640 CAM_HEIGHT=480 CAM_FPS=30
```
