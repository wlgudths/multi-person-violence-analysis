import cv2
import json
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"}


def iter_gcn_results(root):
    root = Path(root)

    for path in sorted(root.glob("*/gcn_result.jsonl")):
        video_id = path.parent.name

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                yield video_id, json.loads(line)

def find_source_video(cfg, video_id):
    source_root = Path(cfg["analysis"]["source_root"])
    path = source_root / f"{video_id}.mp4"

    if path.exists():
        return path

    for path in source_root.rglob(f"{video_id}.*"):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return path

    raise FileNotFoundError(f"Source video not found: {video_id}")

def load_video(builder, video_id):
    for path in builder.result_paths:
        if path.parent.name == video_id:
            return builder.load_result(path)

    raise FileNotFoundError(f"result.jsonl not found: {video_id}")

def get_window_frames(frames, window, window_size):
    start_frame = window["start_frame"]
    start_idx = next((i for i, frame in enumerate(frames) if frame["frame_idx"] == start_frame), None)

    if start_idx is None:
        raise RuntimeError(f"start_frame not found: {start_frame}")

    window_frames = frames[start_idx:start_idx + window_size]

    if len(window_frames) != window_size:
        raise RuntimeError(f"Invalid frame length: {len(window_frames)}")

    return window_frames

def read_source_frame(cap, source_frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame_idx))
    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(f"Failed to read source frame: {source_frame_idx}")

    return frame