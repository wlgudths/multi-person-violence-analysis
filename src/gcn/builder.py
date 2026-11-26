import json
import numpy as np
from pathlib import Path
from mmaction.datasets.transforms import PreNormalize2D


class STGCNBuilder:
    def __init__(self, cfg):
        gcn_cfg = cfg["gcn"]

        self.window_size = gcn_cfg["window_size"]
        self.stride = gcn_cfg["stride"]

        self.min_track_ratio = gcn_cfg["min_track_ratio"]
        self.max_missing_gap = gcn_cfg["max_missing_gap"]

        self.num_keypoints = 17

        self.result_root = Path(cfg["output"]["root_dir"])

        # outputs/<video_id>/result.jsonl
        self.result_paths = sorted(self.result_root.glob("*/result.jsonl"))

        if len(self.result_paths) == 0:
            raise RuntimeError(f"No result.jsonl found in: {self.result_root}")

        self.pre_normalize = PreNormalize2D()

        print(f"Found result videos: {len(self.result_paths)}")

    def __len__(self):
        return len(self.result_paths)
    
    def load_result(self, result_path):
        frames = []

        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                frames.append(json.loads(line))

        return frames

    def iter_results(self):
        for result_path in self.result_paths:

            video_id = result_path.parent.name
            frames = self.load_result(result_path)

            if len(frames) < self.window_size:
                print(
                    f"[SKIP] {video_id} | "
                    f"frames={len(frames)} < T={self.window_size}"
                )
                continue

            yield {
                "video_id": video_id,
                "result_path": result_path,
                "frames": frames,
            }

    def build_windows(self, frames):
        windows = []

        T = self.window_size

        for start in range(
            0,
            len(frames) - T + 1,
            self.stride
        ):
            window_frames = frames[start:start + T]

            sample = self.build_window(
                window_frames
            )

            if sample is not None:
                windows.append(sample)

        return windows

    def build_window(self, frames):
        T = self.window_size

        if len(frames) != T:
            return None

        track_data = {}

        for t, frame in enumerate(frames):
            for person in frame["persons"]:
                track_id = int(person["track_id"])

                if track_id not in track_data:
                    track_data[track_id] = {
                        "observed": np.zeros(T, dtype=bool),
                        "keypoints": np.zeros((T, self.num_keypoints, 2), dtype=np.float32),
                        "scores": np.zeros((T, self.num_keypoints), dtype=np.float32),
                    }

                track = track_data[track_id]
                track["observed"][t] = True
                track["keypoints"][t] = np.asarray(person["keypoints"], dtype=np.float32)
                track["scores"][t] = np.asarray(person["keypoint_scores"], dtype=np.float32)

        valid_track_ids = []

        for track_id, track in track_data.items():
            observed = track["observed"]
            valid_ratio = observed.sum() / T
            max_gap = self._max_missing_gap(observed)

            if valid_ratio >= self.min_track_ratio and max_gap <= self.max_missing_gap:
                valid_track_ids.append(track_id)

        if len(valid_track_ids) == 0:
            return None

        valid_track_ids = sorted(valid_track_ids)
        M = len(valid_track_ids)

        data = np.zeros((M, T, self.num_keypoints, 3), dtype=np.float32)
        observed_mask = np.zeros((M, T), dtype=bool)
        keypoint = np.zeros((M, T, self.num_keypoints, 2), dtype=np.float32)
        keypoint_score = np.zeros((M, T, self.num_keypoints), dtype=np.float32)
        observed_mask = np.zeros((M, T), dtype=bool)

        for m, track_id in enumerate(valid_track_ids):
            track = track_data[track_id]

            kpts = track["keypoints"].copy()
            scores = track["scores"].copy()
            observed = track["observed"].copy()

            kpts = self._interpolate_keypoints(kpts, observed)

            keypoint[m] = kpts
            keypoint_score[m] = scores
            observed_mask[m] = observed

        results = {
            "keypoint": keypoint,
            "keypoint_score": keypoint_score,
            "img_shape": (frames[0]["height"], frames[0]["width"]),
        }

        results = self.pre_normalize(results)

        keypoint = results["keypoint"]
        keypoint_score = results["keypoint_score"]

        data = np.concatenate([keypoint, keypoint_score[..., None]], axis=-1)

        return {
            "data": data,
            "M": M,
            "track_ids": valid_track_ids,
            "observed_mask": observed_mask,

            "start_frame": int(frames[0]["frame_idx"]),
            "end_frame": int(frames[-1]["frame_idx"]),
            "start_source_frame": int(frames[0]["source_frame_idx"]),
            "end_source_frame": int(frames[-1]["source_frame_idx"]),
            "start_timestamp": float(frames[0]["timestamp"]),
            "end_timestamp": float(frames[-1]["timestamp"]),
        }
        
    def _max_missing_gap(self, observed):
        max_gap = 0
        current_gap = 0

        for is_observed in observed:
            if is_observed:
                current_gap = 0
            else:
                current_gap += 1
                max_gap = max(max_gap, current_gap)

        return max_gap

    def _interpolate_keypoints(self, keypoints, observed):
        observed_idx = np.where(observed)[0]

        if len(observed_idx) == 0:
            return keypoints

        all_idx = np.arange(len(observed))

        for v in range(self.num_keypoints):
            for c in range(2):
                keypoints[:, v, c] = np.interp(
                    all_idx,
                    observed_idx,
                    keypoints[observed_idx, v, c]
                )

        return keypoints