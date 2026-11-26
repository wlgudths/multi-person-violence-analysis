from boxmot.reid.core.reid import ReID
from boxmot.trackers.bbox.botsort.botsort import BotSort


def load_tracker(cfg):
    tracker_cfg = cfg["tracker"]

    reid = ReID(
        weights=tracker_cfg["reid_weights"],
        device=cfg["runtime"]["device"],
        half=tracker_cfg.get("half", False),
    )

    tracker = BotSort(
        reid_model=reid.model,

        track_high_thresh=tracker_cfg["track_high_thresh"],
        track_low_thresh=tracker_cfg["track_low_thresh"],
        new_track_thresh=tracker_cfg["new_track_thresh"],

        track_buffer=tracker_cfg["track_buffer"],
        match_thresh=tracker_cfg["match_thresh"],

        proximity_thresh=tracker_cfg["proximity_thresh"],
        appearance_thresh=tracker_cfg["appearance_thresh"],

        cmc_method=tracker_cfg["cmc_method"],
        frame_rate=int(cfg["data"]["target_fps"]),

        with_reid=True,
    )

    return tracker, reid