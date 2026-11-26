"""
Analyze ST-GCN++ classification performance
by the number of valid persons (M).
"""


import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.argparse import parse_args
from utils.result_reader import iter_gcn_results


def load_results(root):
    rows = []

    for video_id, result in iter_gcn_results(root):
        rows.append({
            "video_id": video_id,
            "window_idx": result["window_idx"],
            "M": result["M"],
            "pred": result["pred"],
            "correct": result["correct"],
            "fight_prob": result["fight_prob"],
        })

    return pd.DataFrame(rows)

def analyze_by_m(df):
    summary = []

    for M, group in df.groupby("M"):
        total = len(group)
        tp = int(group["correct"].sum())
        fn = total - tp

        summary.append({
            "M": M,
            "windows": total,
            "TP": tp,
            "FN": fn,
            "recall": tp / total,
            "mean_fight_prob": group["fight_prob"].mean(),
            "median_fight_prob": group["fight_prob"].median(),
            "std_fight_prob": group["fight_prob"].std(),
        })
    
    return pd.DataFrame(summary).sort_values("M")

def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    root = Path(cfg["output"]["root_dir"])

    df = load_results(root)

    if df.empty:
        raise RuntimeError(f"No GCN results found in: {root}")

    summary = analyze_by_m(df)

    df.to_csv(root / "gcn_windows.csv", index=False)
    summary.to_csv(root / "gcn_by_M.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nTotal videos: {df['video_id'].nunique()}")
    print(f"Total windows: {len(df)}")


if __name__ == "__main__":
    main()