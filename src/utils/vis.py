import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.coco import COCO_EDGES, COCO_NAMES
from utils.result_reader import read_source_frame
from utils.result_writer import reset_png_dir, save_image


def get_color(value):
    value = float(np.clip(value, 0, 1))
    color = cv2.applyColorMap(np.uint8([[value * 255]]), cv2.COLORMAP_JET)
    return tuple(int(x) for x in color[0, 0])

def scale_person(person, scale_x, scale_y):
    keypoints = np.asarray(person["keypoints"], dtype=np.float32).copy()
    bbox = np.asarray(person["bbox"], dtype=np.float32).copy()

    keypoints[:, 0] *= scale_x
    keypoints[:, 1] *= scale_y
    bbox[[0, 2]] *= scale_x
    bbox[[1, 3]] *= scale_y

    return keypoints, bbox

def draw_gradcam(canvas, person, cam_values, track_id, scale_x, scale_y):
    keypoints, bbox = scale_person(person, scale_x, scale_y)
    scores = np.asarray(person["keypoint_scores"], dtype=np.float32)

    h, w = canvas.shape[:2]
    line_thickness = max(2, int(w / 900))
    point_size = max(4, int(w / 500))
    font_scale = max(0.7, w / 2200)
    text_thickness = max(2, int(w / 1200))

    for i, j in COCO_EDGES:
        if scores[i] <= 0.2 or scores[j] <= 0.2:
            continue

        value = (cam_values[i] + cam_values[j]) / 2
        color = get_color(value)
        p1 = tuple(np.round(keypoints[i]).astype(int))
        p2 = tuple(np.round(keypoints[j]).astype(int))

        cv2.line(canvas, p1, p2, color, line_thickness, cv2.LINE_AA)

    for i, (x, y) in enumerate(keypoints):
        if scores[i] <= 0.2:
            continue

        value = float(cam_values[i])
        color = get_color(value)
        radius = point_size + int(value * point_size)

        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), line_thickness)
    cv2.putText(canvas, f"ID {track_id}", (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                text_thickness, cv2.LINE_AA)

def get_vis_indices(num_frames, num_vis):
    if num_vis <= 0 or num_vis >= num_frames:
        return list(range(num_frames))

    return np.unique(np.linspace(0, num_frames - 1, num_vis, dtype=int)).tolist()

def save_case_frames(case_dir, source_video_path, window_frames, cam, track_ids, num_vis):
    raw_dir = case_dir / "raw_frames"
    gradcam_dir = case_dir / "gradcam_frames"

    reset_png_dir(raw_dir)
    reset_png_dir(gradcam_dir)

    track_to_m = {int(track_id): i for i, track_id in enumerate(track_ids)}
    vis_indices = get_vis_indices(len(window_frames), num_vis)

    cap = cv2.VideoCapture(str(source_video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_video_path}")

    saved_frames = []

    try:
        for t in vis_indices:
            frame_data = window_frames[t]
            source_idx = int(frame_data["source_frame_idx"])
            raw_frame = read_source_frame(cap, source_idx)

            raw_h, raw_w = raw_frame.shape[:2]
            scale_x = raw_w / int(frame_data["width"])
            scale_y = raw_h / int(frame_data["height"])

            filename = f"t_{t:02d}_src_{source_idx:06d}.png"
            raw_path = raw_dir / filename
            gradcam_path = gradcam_dir / filename

            save_image(raw_path, raw_frame)

            canvas = np.zeros((raw_h, raw_w, 3), dtype=np.uint8)

            for person in frame_data["persons"]:
                track_id = int(person["track_id"])

                if track_id not in track_to_m:
                    continue

                m = track_to_m[track_id]
                draw_gradcam(canvas, person, cam[m, t], track_id, scale_x, scale_y)

            save_image(gradcam_path, canvas)

            saved_frames.append({
                "t": int(t),
                "source_frame_idx": source_idx,
                "raw": str(raw_path),
                "gradcam": str(gradcam_path),
            })

    finally:
        cap.release()

    return saved_frames

def save_heatmaps(case_dir, cam, track_ids):
    temporal = cam.mean(axis=2)

    plt.figure(figsize=(10, 3))
    plt.imshow(temporal, aspect="auto", vmin=0, vmax=1)
    plt.yticks(range(len(track_ids)), [f"ID {track_id}" for track_id in track_ids])
    plt.xlabel("Frame in window")
    plt.ylabel("Track")
    plt.title("Temporal Grad-CAM")
    plt.colorbar()
    plt.tight_layout()

    temporal_path = case_dir / "gradcam_temporal.png"
    plt.savefig(temporal_path, dpi=300)
    plt.close()

    heatmap_paths = []

    for m, track_id in enumerate(track_ids):
        plt.figure(figsize=(10, 5))
        plt.imshow(cam[m].T, aspect="auto", origin="lower", vmin=0, vmax=1)
        plt.yticks(range(17), COCO_NAMES)
        plt.xlabel("Frame in window")
        plt.ylabel("Joint")
        plt.title(f"Grad-CAM | Track {track_id}")
        plt.colorbar()
        plt.tight_layout()

        path = case_dir / f"gradcam_track_{track_id}.png"
        plt.savefig(path, dpi=300)
        plt.close()

        heatmap_paths.append(path)

    return temporal_path, heatmap_paths
