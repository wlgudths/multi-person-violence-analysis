import cv2
from pathlib import Path
from .preprocess import resize_keep_aspect

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mpeg", ".mov", ".mkv"}


def load_image(path, cfg):
    path = Path(path)

    frame = cv2.imread(str(path))

    if frame is None:
        raise FileNotFoundError(f"Failed to load image: {path}")

    return frame
    

def iter_video_frames(path, cfg):
    path = Path(path)

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {path.suffix}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0: native_fps = 30.0

    target_fps = float(cfg["data"]["target_fps"])
    target_long = int(cfg["data"]["target_long"])

    effective_fps = min(native_fps, target_fps)

    sample_interval = 1.0 / effective_fps

    source_frame_idx = 0
    sampled_frame_idx = 0
    next_sample_time = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            timestamp = (source_frame_idx / native_fps)

            if timestamp + 1e-9 >= next_sample_time:

                orig_h, orig_w = frame.shape[:2]

                frame, scale = resize_keep_aspect(frame, target_long=target_long)

                h, w = frame.shape[:2]

                yield {
                    "frame": frame,

                    "frame_idx": sampled_frame_idx,

                    "source_frame_idx": source_frame_idx,

                    "timestamp": timestamp,

                    "orig_width": orig_w,
                    "orig_height": orig_h,

                    "width": w,
                    "height": h,

                    "scale": scale,

                    "native_fps": native_fps,
                    "effective_fps": effective_fps,

                    "source": str(path),
                }

                sampled_frame_idx += 1
                next_sample_time += sample_interval

            source_frame_idx += 1

    finally:
        cap.release()