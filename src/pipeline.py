"""
Multi-person skeleton extraction pipeline.

Performs person detection, multi-object tracking, and pose estimation
on input videos and saves frame-level skeleton results for subsequent
ST-GCN++ action recognition.
"""


from data.dataset import GcnDataset
from utils.config import load_config
from utils.argparse import parse_args
from tracker.tracker import load_tracker
from data.loader import iter_video_frames
from detection.detector import load_detector, detect_persons
from utils.result_writer import open_result_file, save_frame_result
from pose.pose import load_pose_model, estimate_pose, build_tracked_poses


def main():
    args = parse_args()

    # cfg
    cfg = load_config(args.cfg)

    # data
    dataset = GcnDataset(cfg, test=False)
    print(f"Dataset videos: {len(dataset)}")

    # detector
    detector = load_detector(cfg)
    print("Detection model loaded.\n")

    # tracker
    tracker, reid = load_tracker(cfg)
    print("Tracker loaded.")

    pose_model = load_pose_model(cfg)

    print("Pose model loaded.")
    print(type(pose_model))

    # inference
    try:
        for video_idx, sample in enumerate(dataset):

            tracker.reset()

            video_id = sample["video_id"]
            video_path = sample["video_path"]

            print(f"[{video_idx + 1}/{len(dataset)}] {video_id}")
            print(f"Path: {video_path}")
            result_file, result_path = open_result_file(cfg, video_id)

            for frame_data in iter_video_frames(video_path, cfg):
                frame = frame_data["frame"]

                # Detection
                detections = detect_persons(detector, frame, score_thr=cfg["detection"]["score_thr"])
                
                # Tracking
                tracks = tracker.update(detections, frame)

                # Pose estimation
                pose_results = estimate_pose(pose_model, frame, tracks)

                # track_id + pose
                tracked_poses = build_tracked_poses(tracks, pose_results)
                
                # Save
                save_frame_result(result_file, frame_data, detections, tracks, tracked_poses)
                
                print(
                    f"frame={frame_data['frame_idx']:05d} | "
                    f"detections={len(detections)} | "
                    f"tracks={len(tracks)} | "
                    f"poses={len(pose_results)}"
                )
    
    finally:
        result_file.close()
    
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()