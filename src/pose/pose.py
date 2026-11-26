import numpy as np
from mmengine.registry import DefaultScope
from mmpose.apis import init_model, inference_topdown


def load_pose_model(cfg):
    pose_model = init_model(
        cfg["pose"]["config_path"],
        cfg["pose"]["checkpoint_path"],
        device=cfg["runtime"]["device"],
    )

    pose_model.eval()

    return pose_model

def estimate_pose(pose_model, frame, tracks):
    if len(tracks) == 0:
        return []

    bboxes = np.asarray(tracks)[:, :4]

    with DefaultScope.overwrite_default_scope("mmpose"):
        pose_results = inference_topdown(
            pose_model,
            frame,
            bboxes=bboxes
        )

    return pose_results

def build_tracked_poses(tracks, pose_results):
    tracked_poses = []

    for track, pose_result in zip(tracks, pose_results):

        pred = pose_result.pred_instances

        tracked_poses.append({
            "track_id": int(track[4]),
            "bbox": np.asarray(track[:4], dtype=np.float32),
            "det_score": float(track[5]),

            "keypoints": np.asarray(
                pred.keypoints[0],
                dtype=np.float32
            ),

            "keypoint_scores": np.asarray(
                pred.keypoint_scores[0],
                dtype=np.float32
            ),
        })

    return tracked_poses
