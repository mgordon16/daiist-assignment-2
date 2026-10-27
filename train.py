"""
Assignment 2: training pipeline (hard-hat compliance on construction sites).

    uv run python main.py train

Start to finish, with no manual steps:

1. Downloads the dataset from Roboflow Universe (key read via api_key.py).
2. Describes it: images per split, instances per class, object size.
3. Finetunes yolo11n from the COCO checkpoint and saves weights/best.pt.
4. Evaluates on the test split: per-class box metrics, and the box confusion
   matrix at the chosen threshold.
5. Scores every test photo and compares the model against the two baselines
   (pretrained COCO model zero-shot, and a trivial "never alert" baseline).
6. Sweeps the confidence threshold with the cost model in costs.py, searches
   the two thresholds (pass / human review / alert), prices the alternatives
   and runs a sensitivity analysis.
7. Saves everything the dashboard needs into results/.

Set SKIP_TRAINING=1 to reuse an existing weights/best.pt (trained with exactly
the settings below) and only rerun steps 1, 2 and 4-7.
"""

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import costs  # noqa: E402
from api_key import load_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
RESULTS = ROOT / "results"
DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"

# Dataset on Roboflow Universe (public, used as is, not forked)
WORKSPACE = "joseph-nelson"
PROJECT = "hard-hat-workers"
VERSION = 5  # raw_HeadHelmetClasses: classes head + helmet, 70/20/10 split

POSITIVE_CLASS = "head"  # a head WITHOUT a hard hat

TRAIN_SETTINGS = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 32,
    "patience": 15,
    "seed": 0,
}
EVAL_IMGSZ = 640
PROVISIONAL_THRESHOLD = 0.25  # YOLO's default, only for the baseline table


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def download_dataset() -> Path:
    from roboflow import Roboflow

    location = DATASETS / f"{PROJECT}-{VERSION}"
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(location)
    )
    return location


def class_names(data_dir: Path) -> list[str]:
    import yaml

    with open(data_dir / "data.yaml") as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def read_labels(image_path: Path) -> list[tuple[int, float, float, float, float]]:
    lp = label_path(image_path)
    if not lp.exists():
        return []
    rows = []
    for line in lp.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            rows.append((int(p[0]), *map(float, p[1:5])))
    return rows


def describe_dataset(data_dir: Path, names: list[str]) -> None:
    rows, sizes = [], []
    for split in ["train", "valid", "test"]:
        images = sorted((data_dir / split / "images").glob("*"))
        counts = {n: 0 for n in names}
        with_class = {n: 0 for n in names}
        for img in images:
            present = set()
            for c, _, _, w, h in read_labels(img):
                counts[names[c]] += 1
                present.add(names[c])
                sizes.append({"split": split, "class": names[c], "rel_side": (w * h) ** 0.5})
            for n in present:
                with_class[n] += 1
        row = {"split": split, "images": len(images)}
        for n in names:
            row[f"instances_{n}"] = counts[n]
            row[f"images_with_{n}"] = with_class[n]
            row[f"pct_images_with_{n}"] = round(100 * with_class[n] / max(len(images), 1), 1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS / "dataset_splits.csv", index=False)

    s = pd.DataFrame(sizes)
    size_table = s.groupby("class")["rel_side"].describe(percentiles=[0.1, 0.5, 0.9]).round(3)
    size_table["pct_small_lt_32px_at_640"] = (
        s.assign(small=s.rel_side < 32 / 640).groupby("class")["small"].mean() * 100
    ).round(1)
    size_table.to_csv(RESULTS / "dataset_object_sizes.csv")
    print(pd.DataFrame(rows).to_string(index=False))
    print(size_table.to_string())


# ----------------------------------------------------------------------------
# 2. Training
# ----------------------------------------------------------------------------
def train(data_dir: Path) -> None:
    from ultralytics import YOLO

    if os.environ.get("SKIP_TRAINING") == "1" and WEIGHTS.exists():
        print(f"SKIP_TRAINING=1: reusing {WEIGHTS.relative_to(ROOT)}")
        return
    model = YOLO("yolo11n.pt")  # COCO-pretrained checkpoint
    model.train(data=str(data_dir / "data.yaml"), project=str(RUNS), name="hardhat_n",
                exist_ok=True, **TRAIN_SETTINGS)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, WEIGHTS)
    print(f"Saved {WEIGHTS.relative_to(ROOT)}")


# ----------------------------------------------------------------------------
# 3. Box-level evaluation on the test split
# ----------------------------------------------------------------------------
def evaluate_boxes(model, data_dir: Path) -> None:
    m = model.val(data=str(data_dir / "data.yaml"), split="test", imgsz=EVAL_IMGSZ,
                  batch=32, plots=True, project=str(RUNS), name="val_test", exist_ok=True)
    names = model.names
    rows = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append({"class": names[int(c)], "precision": p, "recall": r,
                     "AP50": ap50, "mAP50-95": ap})
    rows.append({"class": "all (mean)", "precision": m.box.mp, "recall": m.box.mr,
                 "AP50": m.box.map50, "mAP50-95": m.box.map})
    per_class = pd.DataFrame(rows).round(3)
    per_class.to_csv(RESULTS / "per_class_test.csv", index=False)
    print(per_class.to_string(index=False))



def box_confusion(model, data_dir: Path, t: float) -> None:
    """Box-level confusion matrix at the chosen operating threshold.

    Ultralytics fills the confusion matrix with the validation `conf`, which is
    0.001 by default (right for mAP, useless for a confusion matrix: it counts
    every near-zero box as a detection). A second pass at the chosen threshold
    gives the matrix that matches how the model is actually used.
    """
    m = model.val(data=str(data_dir / "data.yaml"), split="test", imgsz=EVAL_IMGSZ,
                  batch=32, conf=t, plots=True, project=str(RUNS), name="val_test_cm",
                  exist_ok=True)
    names = model.names
    labels = [names[i] for i in range(len(names))] + ["background"]
    cm = pd.DataFrame(m.confusion_matrix.matrix.astype(int),
                      index=[f"pred_{l}" for l in labels],
                      columns=[f"true_{l}" for l in labels])
    cm.to_csv(RESULTS / "confusion_matrix_boxes_test.csv")
    png = Path(m.save_dir) / "confusion_matrix.png"
    if png.exists():
        shutil.copy(png, RESULTS / "confusion_matrix_boxes_test.png")
    print(f"Box confusion matrix at confidence > {t:.3f}:")
    print(cm.to_string())


# ----------------------------------------------------------------------------
# 4. Photo-level scores and baselines
# ----------------------------------------------------------------------------
def score_photos(model, coco, data_dir: Path) -> pd.DataFrame:
    head = [k for k, v in model.names.items() if v == POSITIVE_CLASS][0]
    images = sorted((data_dir / "test" / "images").glob("*"))
    recs = []
    for i in range(0, len(images), 32):
        batch = [str(p) for p in images[i:i + 32]]
        r_ft = model.predict(batch, conf=0.001, imgsz=EVAL_IMGSZ, verbose=False)
        r_co = coco.predict(batch, conf=0.001, imgsz=EVAL_IMGSZ, classes=[0], verbose=False)
        for path, a, b in zip(batch, r_ft, r_co):
            cls = a.boxes.cls.cpu().numpy()
            conf = a.boxes.conf.cpu().numpy()
            recs.append({
                "image": Path(path).name,
                "y_true": int(any(c == head for c, *_ in read_labels(Path(path)))),
                "score_model": float(conf[cls == head].max()) if (cls == head).any() else 0.0,
                "score_coco_person": float(b.boxes.conf.max()) if len(b.boxes) else 0.0,
            })
    scores = pd.DataFrame(recs)
    scores.to_csv(RESULTS / "test_scores.csv", index=False)
    print(f"{len(scores)} test photos, {scores.y_true.sum()} positive "
          f"({scores.y_true.mean():.1%})")
    return scores


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    ranks = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def at_threshold(y: np.ndarray, s: np.ndarray | None, t: float) -> dict:
    pred = np.zeros_like(y) if s is None else (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn), "accuracy": (tp + tn) / len(y)}


def baselines(scores: pd.DataFrame, per_class: pd.DataFrame, a: dict) -> None:
    y = scores.y_true.to_numpy()
    head_ap = per_class.set_index("class").loc[POSITIVE_CLASS]
    t = PROVISIONAL_THRESHOLD

    def min_cost(column):
        pos, neg = costs.split(scores, column)
        return costs.sweep_single(pos, neg, a).cost_per_1000.min()

    rows = [
        {"approach": "Trivial: never alert", "AP50_head_box": 0.0, "mAP50-95_head_box": 0.0,
         **at_threshold(y, None, t), "AP_photo": float(y.mean()), "ROC_AUC_photo": 0.5,
         "min_cost_per_1000": costs.cost_do_nothing(a)},
        {"approach": "COCO yolo11n zero-shot (alert on 'person')",
         "AP50_head_box": 0.0, "mAP50-95_head_box": 0.0,  # COCO has no head/helmet class
         **at_threshold(y, scores.score_coco_person.to_numpy(), t),
         "AP_photo": average_precision(y, scores.score_coco_person.to_numpy()),
         "ROC_AUC_photo": roc_auc(y, scores.score_coco_person.to_numpy()),
         "min_cost_per_1000": min_cost("score_coco_person")},
        {"approach": "Finetuned yolo11n ('head')",
         "AP50_head_box": head_ap["AP50"], "mAP50-95_head_box": head_ap["mAP50-95"],
         **at_threshold(y, scores.score_model.to_numpy(), t),
         "AP_photo": average_precision(y, scores.score_model.to_numpy()),
         "ROC_AUC_photo": roc_auc(y, scores.score_model.to_numpy()),
         "min_cost_per_1000": min_cost("score_model")},
    ]
    table = pd.DataFrame(rows).round(3)
    table.to_csv(RESULTS / "baselines_test.csv", index=False)
    print(table.to_string(index=False))


# ----------------------------------------------------------------------------
# 5. Cost sweep, two thresholds, alternatives, sensitivity
# ----------------------------------------------------------------------------
def choose_operating_point(scores: pd.DataFrame, a: dict) -> dict:
    pos, neg = costs.split(scores)

    curve = costs.sweep_single(pos, neg, a)
    curve.to_csv(RESULTS / "cost_curve.csv", index=False)
    best1 = curve.loc[curve.cost_per_1000.idxmin()]

    grid = costs.search_two(pos, neg, a)
    grid.to_csv(RESULTS / "two_threshold_grid.csv", index=False)
    best2 = costs.best_two(grid, a)

    table = costs.scenarios(pos, neg, a, best1.threshold, best2.low, best2.high)
    table.round(2).to_csv(RESULTS / "scenarios.csv", index=False)
    print(table.round(2).to_string(index=False))

    # Sensitivity: optimal single threshold and saving vs doing nothing
    sens = []
    for factor in [0.1, 0.25, 0.5, 1, 2, 4]:
        for br in [a["BASE_RATE"] / 4, a["BASE_RATE"] / 2, a["BASE_RATE"], a["BASE_RATE"] * 2]:
            c = pd.DataFrame([costs.single_threshold(pos, neg, t, a, a["C_FN"] * factor, br)
                              for t in costs.THRESHOLDS])
            b = c.loc[c.cost_per_1000.idxmin()]
            nothing = 1000 * br * a["C_FN"] * factor + 1000 * (1 - br) * a["C_TN"]
            sens.append({"C_FN": a["C_FN"] * factor, "base_rate": round(br, 4),
                         "optimal_threshold": b.threshold,
                         "saving_vs_nothing_per_1000": round(nothing - b.cost_per_1000, 1)})
    pd.DataFrame(sens).to_csv(RESULTS / "sensitivity.csv", index=False)

    break_even = None
    for c_fn in np.arange(0, a["C_FN"] * 4, max(a["C_FN"] / 400, 0.01)):
        model_cost = min(costs.single_threshold(pos, neg, t, a, c_fn)["cost_per_1000"]
                         for t in costs.THRESHOLDS)
        if 1000 * a["BASE_RATE"] * c_fn + 1000 * (1 - a["BASE_RATE"]) * a["C_TN"] > model_cost + 1e-9:
            break_even = round(float(c_fn), 2)
            break

    plot_cost_curve(curve, best1, a)
    plot_two_threshold_map(grid, best2)

    decision = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "train_settings": {"model": "yolo11n.pt", **TRAIN_SETTINGS},
        "positive_class": POSITIVE_CLASS,
        "assumptions": a,
        "single_threshold": float(best1.threshold),
        "low_threshold": float(best2.low),
        "high_threshold": float(best2.high),
        "c_fn_break_even": break_even,
    }
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    return decision


def plot_cost_curve(curve: pd.DataFrame, best1: pd.Series, a: dict) -> None:
    nothing, everything = costs.cost_do_nothing(a), costs.cost_human_checks_all(a)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(curve.threshold, curve.cost_per_1000, lw=2, label="model alone")
    ax.axvline(best1.threshold, ls="--", c="red",
               label=f"minimum: threshold {best1.threshold:.3f} -> {best1.cost_per_1000:.0f} EUR")
    ax.axhline(nothing, ls=":", c="gray", label=f"do nothing ({nothing:.0f} EUR)")
    ax.axhline(everything, ls="-.", c="purple", label=f"person checks all ({everything:.0f} EUR)")
    ax.set_xlabel("confidence threshold for 'head'")
    ax.set_ylabel("expected cost per 1,000 photos (EUR)")
    ax.set_ylim(0, max(nothing, everything) * 1.15)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "cost_curve.png", dpi=120); plt.close(fig)


def plot_two_threshold_map(grid: pd.DataFrame, best2: pd.Series) -> None:
    piv = grid.pivot(index="high", columns="low", values="cost_per_1000")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis",
                   extent=[costs.GRID[0], costs.GRID[-1], costs.GRID[0], costs.GRID[-1]])
    ax.scatter(best2.low, best2.high, c="red", s=80, marker="x",
               label=f"chosen ({best2.low:.2f}, {best2.high:.2f})")
    ax.set_xlabel("low threshold (below: pass)"); ax.set_ylabel("high threshold (above: alert)")
    fig.colorbar(im, label="EUR per 1,000 photos"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "two_threshold_map.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------
# 6. Failure examples at the chosen operating point
# ----------------------------------------------------------------------------
def save_failures(model, scores: pd.DataFrame, data_dir: Path, t: float) -> None:
    from PIL import Image

    head = [k for k, v in model.names.items() if v == POSITIVE_CLASS][0]
    fn = scores[(scores.y_true == 1) & (scores.score_model < t)].sort_values("score_model")
    fp = scores[(scores.y_true == 0) & (scores.score_model >= t)].sort_values(
        "score_model", ascending=False)
    for df, name, title in [(fn.head(6), "failures_false_negatives.png",
                             "Missed: someone bare-headed, no alert"),
                            (fp.head(6), "failures_false_positives.png",
                             "False alarm: alert, nobody labelled bare-headed")]:
        if df.empty:
            continue
        fig, axes = plt.subplots(2, len(df), figsize=(4 * len(df), 7.5), squeeze=False)
        for j, (_, row) in enumerate(df.iterrows()):
            path = data_dir / "test" / "images" / row.image
            pred = model.predict(str(path), conf=t, verbose=False)[0].plot()[:, :, ::-1]
            axes[0, j].imshow(pred); axes[0, j].set_title(f"model, head score {row.score_model:.2f}")
            gt = np.array(Image.open(path).convert("RGB"))
            H, W = gt.shape[:2]
            axes[1, j].imshow(gt); axes[1, j].set_title("ground truth (red = head)")
            for c, x, y, w, h in read_labels(path):
                axes[1, j].add_patch(plt.Rectangle(((x - w / 2) * W, (y - h / 2) * H), w * W, h * H,
                                                   fill=False, lw=2,
                                                   ec="red" if c == head else "lime"))
            axes[0, j].axis("off"); axes[1, j].axis("off")
        fig.suptitle(f"{title} (threshold {t:.2f})")
        fig.tight_layout(); fig.savefig(RESULTS / name, dpi=100); plt.close(fig)
    print(f"At threshold {t:.3f}: {len(fn)} false negatives, {len(fp)} false positives (test)")


# ----------------------------------------------------------------------------
def main() -> None:
    from ultralytics import YOLO

    RESULTS.mkdir(exist_ok=True)
    data_dir = download_dataset()
    names = class_names(data_dir)
    print("Classes:", names)
    describe_dataset(data_dir, names)

    train(data_dir)

    model = YOLO(str(WEIGHTS))
    coco = YOLO("yolo11n.pt")
    evaluate_boxes(model, data_dir)
    scores = score_photos(model, coco, data_dir)
    per_class = pd.read_csv(RESULTS / "per_class_test.csv")
    a = dict(costs.ASSUMPTIONS)
    baselines(scores, per_class, a)
    decision = choose_operating_point(scores, a)
    box_confusion(model, data_dir, decision["single_threshold"])
    decision["box_confusion_threshold"] = decision["single_threshold"]
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2))
    save_failures(model, scores, data_dir, decision["single_threshold"])
    print(f"\nDone. Weights in {WEIGHTS.relative_to(ROOT)}, results in {RESULTS.name}/")


if __name__ == "__main__":
    main()
