"""
Margin-based Top-2 analysis for multi-person
skeleton-based violence recognition.

Compare uniform person aggregation (All-M) with
aggregation of the two persons with the highest
Fight logit margins.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from utils.config import load_config
from utils.argparse import parse_args
from utils.metrics import softmax, logit_margin
from utils.result_reader import iter_gcn_results


def select_top2(person_logits, fight_class):
    nonfight_class = 1 - fight_class

    margins = logit_margin(person_logits, fight_class, nonfight_class)

    indices = np.argsort(margins)[::-1][:2]

    logits = person_logits[indices].mean(axis=0)
    probs = softmax(logits)

    return {
        "indices": indices,
        "person_margins": margins,
        "logits": logits,
        "probs": probs,
        "pred": int(np.argmax(probs)),
        "margin": float(logit_margin(logits, fight_class, nonfight_class)),
        "fight_prob": float(probs[fight_class]),
    }

def analyze_windows(root, fight_class):
    nonfight_class = 1 - fight_class
    rows = []

    for video_id, result in iter_gcn_results(root):
        if result["M"] < 2:
            continue

        person_logits = np.asarray([p["logits"] for p in result["persons"]], dtype=np.float32,)
        track_ids = [p["track_id"] for p in result["persons"]]
        top2 = select_top2(person_logits, fight_class)
        all_logits = np.asarray(result["logits"], dtype=np.float32)
        all_margin = float(logit_margin(all_logits, fight_class, nonfight_class))
        all_pred = result["pred"]
        all_prob = result["fight_prob"]

        rows.append({
            "video_id": video_id,
            "window_idx": result["window_idx"],
            "start_timestamp": result["start_timestamp"],
            "M": result["M"],

            "all_pred": all_pred,
            "all_fight_prob": all_prob,
            "all_margin": all_margin,

            "top2_pred": top2["pred"],
            "top2_fight_prob": top2["fight_prob"],
            "top2_margin": top2["margin"],

            "margin_gain": top2["margin"] - all_margin,
            "prob_gain": top2["fight_prob"] - all_prob,

            "rescued": (all_pred != fight_class and top2["pred"] == fight_class),
            "top2_track_ids": ",".join(str(track_ids[i]) for i in top2["indices"]),
            "person_margins": ",".join(f"{m:.6f}" for m in top2["person_margins"]),
        })

    return pd.DataFrame(rows)

def analyze_by_m(df, fight_class):
    summary = []

    for M, group in df.groupby("M"):
        total = len(group)

        all_tp = int((group["all_pred"] == fight_class).sum())
        top2_tp = int((group["top2_pred"] == fight_class).sum())

        all_fn = total - all_tp
        top2_fn = total - top2_tp

        all_recall = all_tp / total
        top2_recall = top2_tp / total

        rescued_fn = int(group["rescued"].sum())

        summary.append({
            "M": M,
            "windows": total,

            "all_TP": all_tp,
            "all_FN": all_fn,
            "all_recall": all_recall,
            "all_mean_prob": group["all_fight_prob"].mean(),

            "top2_TP": top2_tp,
            "top2_FN": top2_fn,
            "top2_recall": top2_recall,
            "top2_mean_prob": group["top2_fight_prob"].mean(),

            "recall_gain_pp": (top2_recall - all_recall) * 100,

            "mean_prob_gain": group["prob_gain"].mean(),
            "mean_margin_gain": group["margin_gain"].mean(),

            "rescued_FN": rescued_fn,
            "rescue_rate": (rescued_fn / all_fn if all_fn > 0 else 0),
        })

    return pd.DataFrame(summary).sort_values("M")

def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    root = Path(cfg["output"]["root_dir"])
    fight_class = cfg["gcn"]["fight_class"]

    df = analyze_windows(root, fight_class)

    if df.empty:
        raise RuntimeError(f"No valid multi-person GCN results found in: {root}")

    summary = analyze_by_m(df, fight_class)

    df.to_csv(root / "top2_window_analysis.csv", index=False)
    summary.to_csv(root / "top2_by_M.csv", index=False)

    print(summary.to_string(index=False))

    print(f"\nTotal videos: {df['video_id'].nunique()}")
    print(f"Total windows: {len(df)}")


if __name__ == "__main__":
    main()