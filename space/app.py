"""Gradio app for the Grabette data pipeline."""

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import gradio as gr

WORK_DIR = Path(os.environ.get("GRABETTE_WORK_DIR", "/tmp/grabette-work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS_DIR = Path(__file__).parent / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zip_episode(ep_dir: Path, exclude: set[str] | None = None) -> Path:
    """Zip all files in ep_dir (excluding raw video) and return zip path."""
    exclude = exclude or {"raw_video.mp4"}
    zip_path = WORK_DIR / f"{ep_dir.name}_slam.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ep_dir.rglob("*"):
            if f.is_file() and f.name not in exclude:
                zf.write(f, f.relative_to(ep_dir))
    return zip_path


def _stream(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


# ---------------------------------------------------------------------------
# Step 1: Create Map
# ---------------------------------------------------------------------------

def run_create_map(video_file, imu_file, retries, deterministic):
    """Run two-pass SLAM and stream logs. Yields (logs, download_path, ep_dir)."""
    if video_file is None or imu_file is None:
        yield "Please upload both raw_video.mp4 and imu_data.json.", None, None
        return

    ep_dir = Path(tempfile.mkdtemp(dir=WORK_DIR, prefix="ep_"))
    shutil.copy(video_file, ep_dir / "raw_video.mp4")
    shutil.copy(imu_file, ep_dir / "imu_data.json")

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "create_map.py"),
        "-i", str(ep_dir),
        "--retries", str(int(retries)),
    ]
    if deterministic:
        cmd.append("--deterministic")

    logs = ""
    proc = _stream(cmd)
    for line in proc.stdout:
        logs += line
        yield logs, None, None
    proc.wait()

    if proc.returncode == 0:
        zip_path = _zip_episode(ep_dir)
        yield logs + "\nDone!\n", str(zip_path), str(ep_dir)
    else:
        yield logs + "\nFailed — check the logs above.\n", None, None


# ---------------------------------------------------------------------------
# Step 2: Batch SLAM
# ---------------------------------------------------------------------------

def run_batch_slam(episodes_zip, map_file, num_workers, max_lost_frames):
    """Localize multiple episodes against an existing map."""
    if episodes_zip is None or map_file is None:
        yield "Please upload both an episodes ZIP and a map file (.osa).", None
        return

    work = Path(tempfile.mkdtemp(dir=WORK_DIR, prefix="batch_"))
    episodes_dir = work / "episodes"
    episodes_dir.mkdir()

    # Extract episodes zip
    with zipfile.ZipFile(episodes_zip, "r") as zf:
        zf.extractall(episodes_dir)

    shutil.copy(map_file, work / "map_atlas.osa")

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "batch_slam.py"),
        "-i", str(episodes_dir),
        "-m", str(work / "map_atlas.osa"),
        "-n", str(int(num_workers)),
        "--max_lost_frames", str(int(max_lost_frames)),
    ]

    logs = ""
    proc = _stream(cmd)
    for line in proc.stdout:
        logs += line
        yield logs, None
    proc.wait()

    if proc.returncode == 0:
        zip_path = WORK_DIR / f"{work.name}_batch_results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in episodes_dir.rglob("camera_trajectory.csv"):
                zf.write(f, f.relative_to(episodes_dir))
        yield logs + "\nDone!\n", str(zip_path)
    else:
        yield logs + "\nFailed — check the logs above.\n", None


# ---------------------------------------------------------------------------
# Step 3: Generate Dataset
# ---------------------------------------------------------------------------

def run_generate_dataset(ep_dir_state, repo_id, task, fps):
    """Convert SLAM outputs into a LeRobot v3 dataset."""
    if not ep_dir_state:
        yield "Run Create Map first (step 1).", None
        return
    if not repo_id.strip():
        yield "Please provide a dataset repo ID (e.g. myuser/grabette-demo).", None
        return

    ep_dir = Path(ep_dir_state)
    dataset_root = WORK_DIR / "datasets"
    dataset_root.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "generate_dataset.py"),
        "-i", str(ep_dir.parent),   # parent dir containing episode subdirs
        "--repo_id", repo_id.strip(),
        "--task", task.strip() or "grabette episode",
        "--fps", str(float(fps)),
        "--root", str(dataset_root),
    ]

    logs = ""
    proc = _stream(cmd)
    for line in proc.stdout:
        logs += line
        yield logs, None
    proc.wait()

    if proc.returncode == 0:
        # Zip the generated dataset
        dataset_dir = dataset_root / repo_id.replace("/", "--")
        if not dataset_dir.exists():
            # Fallback: zip everything in dataset_root
            dataset_dir = dataset_root
        zip_path = WORK_DIR / f"dataset_{repo_id.replace('/', '_')}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in dataset_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(dataset_dir))
        yield logs + "\nDone!\n", str(zip_path)
    else:
        yield logs + "\nFailed — check the logs above.\n", None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Grabette Data Pipeline") as demo:
    gr.Markdown("# Grabette Data Pipeline")
    gr.Markdown(
        "Process raw GRABETTE episodes through ORB-SLAM3 "
        "and export [LeRobot v3](https://huggingface.co/docs/lerobot) datasets."
    )

    # Shared state: path to the episode directory produced by step 1
    ep_dir_state = gr.State(None)

    with gr.Tab("1 — Create Map"):
        gr.Markdown(
            "Upload a raw episode and run two-pass SLAM to produce "
            "a camera trajectory and map."
        )
        with gr.Row():
            video_input = gr.File(label="raw_video.mp4", file_types=[".mp4"])
            imu_input = gr.File(label="imu_data.json", file_types=[".json"])
        with gr.Row():
            retries = gr.Slider(0, 5, value=3, step=1, label="Pass-1 retries")
            deterministic = gr.Checkbox(label="Deterministic mode (slower, reproducible)")
        slam_btn = gr.Button("Run SLAM", variant="primary")
        slam_logs = gr.Textbox(label="Logs", lines=20, max_lines=40, autoscroll=True)
        slam_download = gr.File(label="Download SLAM results (.zip)")

        slam_btn.click(
            fn=run_create_map,
            inputs=[video_input, imu_input, retries, deterministic],
            outputs=[slam_logs, slam_download, ep_dir_state],
        )

    with gr.Tab("2 — Batch SLAM"):
        gr.Markdown(
            "Localize multiple episodes against an existing map. "
            "Upload a ZIP of episode directories and the map `.osa` file."
        )
        with gr.Row():
            episodes_zip = gr.File(label="Episodes ZIP", file_types=[".zip"])
            map_file = gr.File(label="map_atlas.osa", file_types=[".osa"])
        with gr.Row():
            num_workers = gr.Slider(1, 8, value=2, step=1, label="Parallel workers")
            max_lost_frames = gr.Slider(0, 200, value=60, step=10, label="Max lost frames")
        batch_btn = gr.Button("Run Batch SLAM", variant="primary")
        batch_logs = gr.Textbox(label="Logs", lines=20, max_lines=40, autoscroll=True)
        batch_download = gr.File(label="Download trajectories (.zip)")

        batch_btn.click(
            fn=run_batch_slam,
            inputs=[episodes_zip, map_file, num_workers, max_lost_frames],
            outputs=[batch_logs, batch_download],
        )

    with gr.Tab("3 — Generate Dataset"):
        gr.Markdown(
            "Convert SLAM outputs from step 1 into a LeRobot v3 dataset. "
            "Run step 1 first, or the episode directory state will be empty."
        )
        repo_id = gr.Textbox(label="Dataset repo ID", placeholder="myuser/grabette-demo")
        task = gr.Textbox(label="Task description", placeholder="cup manipulation")
        fps = gr.Number(label="FPS", value=46.0)
        dataset_btn = gr.Button("Generate Dataset", variant="primary")
        dataset_logs = gr.Textbox(label="Logs", lines=20, max_lines=40, autoscroll=True)
        dataset_download = gr.File(label="Download dataset (.zip)")

        dataset_btn.click(
            fn=run_generate_dataset,
            inputs=[ep_dir_state, repo_id, task, fps],
            outputs=[dataset_logs, dataset_download],
        )

demo.queue().launch(server_name="0.0.0.0", server_port=7860)
