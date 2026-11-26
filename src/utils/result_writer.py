import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.coco import COCO_NAMES


def open_result_file(cfg, video_id):
    output_dir = Path(cfg["output"]["root_dir"]) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "result.jsonl"

    file = open(
        result_path,
        "w",
        encoding="utf-8"
    )

    return file, result_path

def save_frame_result(
    file,
    frame_data,
    detections,
    tracks,
    tracked_poses,
):
    result = {
        "frame_idx": int(frame_data["frame_idx"]),
        "source_frame_idx": int(frame_data["source_frame_idx"]),
        "timestamp": float(frame_data["timestamp"]),

        "width": int(frame_data["width"]),
        "height": int(frame_data["height"]),

        "detections": detections.tolist(),

        "tracks": tracks.tolist(),

        "persons": [
            {
                "track_id": int(person["track_id"]),
                "bbox": person["bbox"].tolist(),
                "det_score": float(person["det_score"]),
                "keypoints": person["keypoints"].tolist(),
                "keypoint_scores": person["keypoint_scores"].tolist(),
            }
            for person in tracked_poses
        ],
    }

    file.write(
        json.dumps(
            result,
            ensure_ascii=False
        ) + "\n"
    )

def open_gcn_result_file(cfg, video_id):
    output_dir = Path(cfg["output"]["root_dir"]) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "gcn_result.jsonl"
    return open(path, "w", encoding="utf-8"), path

def save_gcn_result(file, result):
    file.write(json.dumps(result, ensure_ascii=False) + "\n")

def build_gcn_result(window_idx, window, inference, fight_class):
    return {
        "window_idx": window_idx,

        "start_frame": window["start_frame"],
        "end_frame": window["end_frame"],

        "start_source_frame": window["start_source_frame"],
        "end_source_frame": window["end_source_frame"],

        "start_timestamp": window["start_timestamp"],
        "end_timestamp": window["end_timestamp"],

        "M": window["M"],
        "track_ids": window["track_ids"],
        "observed_mask": window["observed_mask"].tolist(),

        "gt": fight_class,
        "pred": inference["pred"],
        "correct": inference["pred"] == fight_class,

        "logits": inference["logits"],
        "probs": inference["probs"],
        "fight_prob": inference["fight_prob"],

        "persons": inference["persons"],
    }

def reset_png_dir(path):
    path.mkdir(parents=True, exist_ok=True)

    for file in path.glob("*.png"):
        file.unlink()

def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")

def build_person_results(window, persons, top2, cam):
    results = []

    for i, track_id in enumerate(window["track_ids"]):
        cam_max = float(cam[i].max())

        if cam_max > 1e-6:
            peak_t, peak_joint = np.unravel_index(np.argmax(cam[i]), cam[i].shape)
            peak_t = int(peak_t)
            peak_joint = int(peak_joint)
            peak_joint_name = COCO_NAMES[peak_joint]
        else:
            peak_t = None
            peak_joint = None
            peak_joint_name = None

        results.append({
            "track_id": int(track_id),
            "top2_selected": i in top2["indices"],
            "fight_prob": persons[i]["fight_prob"],
            "margin": persons[i]["margin"],
            "cam_mean": float(cam[i].mean()),
            "cam_max": cam_max,
            "cam_peak_frame": peak_t,
            "cam_peak_joint": peak_joint,
            "cam_peak_joint_name": peak_joint_name,
        })

    return results

def save_person_table(case_dir, person_results):
    columns = [
        "track_id",
        "top2_selected",
        "fight_prob",
        "margin",
        "cam_mean",
        "cam_max",
        "cam_peak_frame",
        "cam_peak_joint",
        "cam_peak_joint_name",
    ]

    path = case_dir / "person_scores.csv"
    pd.DataFrame(person_results)[columns].to_csv(path, index=False)

    return path

def build_case_summary(
    video_id, source_video_path, window_idx, window,
    all_pred, all_fight_prob, logits, top2, fight_class,
    person_results, saved_frames, case_dir, person_table_path,
    temporal_path, heatmap_paths,
):
    summary = {
        "video_id": video_id,
        "source_video": str(source_video_path),
        "window_idx": int(window_idx),
        "M": int(window["M"]),
        "start_frame": int(window["start_frame"]),
        "end_frame": int(window["end_frame"]),
        "start_source_frame": int(window["start_source_frame"]),
        "end_source_frame": int(window["end_source_frame"]),
        "start_timestamp": float(window["start_timestamp"]),
        "end_timestamp": float(window["end_timestamp"]),
        "track_ids": [int(x) for x in window["track_ids"]],

        "all_pred": all_pred,
        "all_fight_prob": all_fight_prob,
        "all_logits": logits[0].cpu().tolist(),

        "top2_track_ids": [int(window["track_ids"][i]) for i in top2["indices"]],
        "top2_pred": top2["pred"],
        "top2_fight_prob": top2["fight_prob"],
        "top2_logits": top2["logits"].tolist(),

        "prob_gain": top2["fight_prob"] - all_fight_prob,
        "rescued": all_pred != fight_class and top2["pred"] == fight_class,

        "persons": person_results,
        "saved_frames": saved_frames,

        "files": {
            "raw_frames_dir": str(case_dir / "raw_frames"),
            "gradcam_frames_dir": str(case_dir / "gradcam_frames"),
            "person_scores": str(person_table_path),
            "gradcam_temporal": str(temporal_path),
            "gradcam_tracks": [str(path) for path in heatmap_paths],
        },
    }

    return summary

def save_case_summary(case_dir, summary):
    path = case_dir / "case_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path
