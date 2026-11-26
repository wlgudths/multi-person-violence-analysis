"""
Run ST-GCN++ inference on Stable-M skeleton windows.

Save All-M and person-wise classification results
for subsequent aggregation analysis.
"""


from gcn.gcn import load_stgcn
from utils.config import load_config
from gcn.builder import STGCNBuilder
from utils.argparse import parse_args
from gcn.inference import infer_window
from utils.result_writer import open_gcn_result_file, build_gcn_result, save_gcn_result


def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    builder = STGCNBuilder(cfg)
    model = load_stgcn(cfg)

    device = cfg["runtime"]["device"]
    fight_class = cfg["gcn"]["fight_class"]

    for video_data in builder.iter_results():
        video_id = video_data["video_id"]
        windows = builder.build_windows(video_data["frames"])

        if not windows:
            continue

        file, path = open_gcn_result_file(cfg, video_id)

        with file:
            for window_idx, window in enumerate(windows):
                inference = infer_window(model, window, device, fight_class)

                result = build_gcn_result(window_idx, window, inference, fight_class)

                save_gcn_result(file, result)

        print(
            f"[GCN] {video_id} | "
            f"windows={len(windows)} | saved={path}"
            )


if __name__ == "__main__":
    main()