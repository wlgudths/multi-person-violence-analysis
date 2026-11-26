"""
Qualitative case analysis for multi-person violence recognition.

Analyze selected cases using person-wise Fight logit margins,
Margin-based Top-2 aggregation, and spatiotemporal Grad-CAM.
"""


import pandas as pd
from pathlib import Path
from gcn.gcn import load_stgcn
from gcn.builder import STGCNBuilder
from utils.config import load_config
from utils.argparse import parse_args
from utils.vis import save_case_frames, save_heatmaps
from gcn.inference import analyze_persons, analyze_top2
from gcn.gradcam import build_input, forward_gradcam, upsample_cam
from utils.result_reader import find_source_video, load_video, get_window_frames
from utils.result_writer import build_person_results, save_person_table, build_case_summary, save_case_summary


def select_cases(cfg, root):
    analysis_cfg = cfg["analysis"]

    if not analysis_cfg.get("auto_select", True):
        return [{
            "video_id": analysis_cfg["video_id"],
            "window_idx": int(analysis_cfg["window_idx"]),
            "M": None,
            "margin_gain": None,
        }]

    csv_path = root / "top2_window_analysis.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["rescued"].astype(str).str.lower() == "true"].copy()

    target_M = analysis_cfg.get("target_M", [3, 4])
    if isinstance(target_M, int):
        target_M = [target_M]

    top_k = int(analysis_cfg.get("top_k", 3))
    cases = []

    for M in target_M:
        group = df[df["M"] == M].sort_values("margin_gain", ascending=False).head(top_k)

        for _, row in group.iterrows():
            cases.append({
                "video_id": row["video_id"],
                "window_idx": int(row["window_idx"]),
                "M": int(row["M"]),
                "margin_gain": float(row["margin_gain"]),
            })

    if not cases:
        raise RuntimeError("No rescued FN cases found.")

    return cases

def analyze_case(cfg, builder, model, case):
    root = Path(cfg["output"]["root_dir"])
    device = cfg["runtime"]["device"]
    fight_class = cfg["gcn"]["fight_class"]
    num_vis = int(cfg["analysis"].get("num_vis_frames", 0))

    video_id = case["video_id"]
    window_idx = case["window_idx"]

    source_video_path = find_source_video(cfg, video_id)

    frames = load_video(builder, video_id)
    windows = builder.build_windows(frames)

    if window_idx >= len(windows):
        raise RuntimeError(f"Invalid window_idx: {video_id} | {window_idx}/{len(windows)}")

    window = windows[window_idx]
    window_frames = get_window_frames(frames, window, builder.window_size)

    case_name = f"M{window['M']}_{video_id}_w{window_idx:04d}"
    case_dir = root / "_cases" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    x = build_input(window, device)
    feat, logits, probs, cam = forward_gradcam(model, x, fight_class)

    persons = analyze_persons(model, feat, fight_class)
    top2 = analyze_top2(model, feat, persons, fight_class)

    cam = upsample_cam(cam, window["data"].shape[1])
    person_results = build_person_results(window, persons, top2, cam)

    all_pred = int(probs.argmax(dim=1).item())
    all_fight_prob = float(probs[0, fight_class])

    saved_frames = save_case_frames(
        case_dir,
        source_video_path,
        window_frames,
        cam,
        window["track_ids"],
        num_vis,
    )

    temporal_path, heatmap_paths = save_heatmaps(case_dir, cam, window["track_ids"])
    person_table_path = save_person_table(case_dir, person_results)

    summary = build_case_summary(
        video_id, source_video_path, window_idx, window,
        all_pred, all_fight_prob, logits, top2, fight_class,
        person_results, saved_frames, case_dir, person_table_path,
        temporal_path, heatmap_paths,
    )

    save_case_summary(case_dir, summary)

    print(f"[CASE] M={window['M']} | {video_id} | window={window_idx}")
    print(f"All-M : pred={all_pred} | fight_prob={all_fight_prob:.4f}")
    print(f"Top-2 : pred={top2['pred']} | fight_prob={top2['fight_prob']:.4f}")
    print(f"Top-2 IDs: {summary['top2_track_ids']}")

    for person in person_results:
        peak = (
            f"{person['cam_peak_joint_name']}@{person['cam_peak_frame']}"
            if person["cam_peak_joint_name"] is not None
            else "no positive activation"
        )

        print(
            f"ID {person['track_id']} | margin={person['margin']:.4f} | "
            f"fight_prob={person['fight_prob']:.4f} | "
            f"top2={person['top2_selected']} | CAM={peak}"
        )

    print(f"Saved frames: {len(saved_frames)}")
    print(f"Saved: {case_dir}")

    return {
        "video_id": video_id,
        "window_idx": int(window_idx),
        "M": int(window["M"]),
        "all_fight_prob": all_fight_prob,
        "top2_fight_prob": top2["fight_prob"],
        "prob_gain": top2["fight_prob"] - all_fight_prob,
        "rescued": summary["rescued"],
        "case_dir": str(case_dir),
    }

def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    root = Path(cfg["output"]["root_dir"])
    cases_root = root / "_cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    builder = STGCNBuilder(cfg)
    model = load_stgcn(cfg)

    cases = select_cases(cfg, root)

    print(f"Selected cases: {len(cases)}")

    results = []

    for i, case in enumerate(cases):
        print(f"\n[{i + 1}/{len(cases)}] {case['video_id']} | window={case['window_idx']}")

        try:
            results.append(analyze_case(cfg, builder, model, case))
        except Exception as e:
            print(f"[ERROR] {case['video_id']} | {e}")

    if results:
        summary_path = cases_root / "cases_summary.csv"
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()