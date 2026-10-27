"""
Assignment 2: Gradio dashboard for the hard-hat compliance model.

    uv run python main.py app

Never trains anything. It loads the committed weights in weights/ and the
outputs train.py saved in results/, and recomputes costs live with costs.py
(the same arithmetic train.py used to choose the operating point).
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import ImageDraw, ImageFont

import costs

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
RESULTS = ROOT / "results"

for needed in [WEIGHTS, RESULTS / "decision.json", RESULTS / "test_scores.csv"]:
    if not needed.exists():
        raise SystemExit(f"Missing {needed.relative_to(ROOT)}. Run `uv run python main.py train` first.")

DECISION = json.loads((RESULTS / "decision.json").read_text())
A = DECISION["assumptions"]
SCORES = pd.read_csv(RESULTS / "test_scores.csv")
POS, NEG = costs.split(SCORES)
CURVE = costs.sweep_single(POS, NEG, A)
T_SINGLE = DECISION["single_threshold"]
LOW0, HIGH0 = DECISION["low_threshold"], DECISION["high_threshold"]

from ultralytics import YOLO  # noqa: E402  (after the file checks, so errors are readable)

MODEL = YOLO(str(WEIGHTS))
NAMES = MODEL.names
HEAD = next((k for k, v in NAMES.items() if v == DECISION["positive_class"]), 0)


def read_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def image_or_none(name: str) -> str | None:
    path = RESULTS / name
    return str(path) if path.exists() else None


# ----------------------------------------------------------------------------
# Cost view
# ----------------------------------------------------------------------------
def ordered(low: float, high: float) -> tuple[float, float]:
    return (low, high) if low <= high else (high, low)


def matrices(low: float, high: float):
    r = costs.two_thresholds(POS, NEG, low, high, A)
    m = r["matrix_per_1000"]
    expected = pd.DataFrame(
        {"pass": [m["pos"]["pass"], m["neg"]["pass"]],
         "human review": [m["pos"]["review"], m["neg"]["review"]],
         "alert": [m["pos"]["alert"], m["neg"]["alert"]]},
        index=["someone bare-headed", "nobody bare-headed"]).round(1)
    expected.insert(0, "real (per 1,000 photos)", expected.index)
    expected = expected.reset_index(drop=True)

    raw = pd.DataFrame(
        {"pass": [int((POS < low).sum()), int((NEG < low).sum())],
         "human review": [int(((POS >= low) & (POS < high)).sum()),
                          int(((NEG >= low) & (NEG < high)).sum())],
         "alert": [int((POS >= high).sum()), int((NEG >= high).sum())]},
        index=[f"someone bare-headed (n={len(POS)})", f"nobody bare-headed (n={len(NEG)})"])
    raw.insert(0, "real (test split counts)", raw.index)
    raw = raw.reset_index(drop=True)
    return r, expected, raw


def cost_figure(low: float, high: float, current_cost: float) -> Figure:
    fig = Figure(figsize=(9, 4.6))
    ax = fig.subplots()
    nothing, everything = costs.cost_do_nothing(A), costs.cost_human_checks_all(A)
    ax.plot(CURVE.threshold, CURVE.cost_per_1000, lw=2, label="model alone (single threshold)")
    best = CURVE.loc[CURVE.cost_per_1000.idxmin()]
    ax.axvline(T_SINGLE, ls="--", c="red",
               label=f"chosen single threshold {T_SINGLE:.3f} -> {best.cost_per_1000:.0f} EUR")
    ax.axvspan(low, high, color="orange", alpha=0.15, label=f"review band now [{low:.2f}, {high:.2f})")
    ax.axhline(current_cost, c="green", lw=1.5, label=f"current two-threshold design {current_cost:.0f} EUR")
    ax.axhline(nothing, ls=":", c="gray", label=f"do nothing {nothing:.0f} EUR")
    ax.axhline(everything, ls="-.", c="purple", label=f"person checks all {everything:.0f} EUR")
    ax.set_xlabel("confidence threshold for 'head' (no hard hat)")
    ax.set_ylabel("expected cost per 1,000 photos (EUR)")
    ax.set_ylim(0, max(nothing, everything) * 1.15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    return fig


def update_costs(low: float, high: float):
    low, high = ordered(low, high)
    r, expected, raw = matrices(low, high)
    k = A["CASES_PER_DAY"] / 1000
    kpis = (
        f"### Expected cost: **{r['cost_per_1000']:.1f} EUR per 1,000 photos** "
        f"({r['cost_per_1000'] * k:.0f} EUR/day)\n"
        f"- Sent to a person: **{r['share_review']:.1%}** of photos = "
        f"**{r['reviews_per_day']:.0f} reviews/day**, "
        f"{r['reviews_per_day'] * A['REVIEW_SECONDS'] / 3600:.1f} h/day = "
        f"**{r['people']:.2f} full-time reviewers** (limit {A['MAX_REVIEWERS']})\n"
        f"- Bare-headed photos that escape: **{r['FN_1000'] * k:.2f}/day** · "
        f"false alarms reaching the site manager: **{r['FP_1000'] * k:.2f}/day**"
    )
    if r["people"] > A["MAX_REVIEWERS"]:
        kpis += "\n\n⚠️ This band needs more reviewers than the company would pay for."
    alternatives = costs.scenarios(POS, NEG, A, T_SINGLE, low, high).round(2)
    return kpis, expected, raw, cost_figure(low, high, r["cost_per_1000"]), alternatives


# ----------------------------------------------------------------------------
# Image view
# ----------------------------------------------------------------------------
def detect(image):
    if image is None:
        return []
    r = MODEL.predict(image, conf=0.01, imgsz=640, verbose=False)[0]
    return [(int(c), float(cf), *map(float, xyxy))
            for c, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                   r.boxes.xyxy.tolist())]


def render(image, boxes, low: float, high: float):
    if image is None:
        return None, "Upload a photo to see the model's predictions."
    low, high = ordered(low, high)
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(12, img.width // 45))
    except TypeError:
        font = ImageFont.load_default()
    floor = max(low, 0.05)  # do not draw near-zero boxes even when low = 0
    score = max((cf for c, cf, *_ in boxes if c == HEAD), default=0.0)
    for c, cf, x1, y1, x2, y2 in boxes:
        if cf < floor:
            continue
        if c == HEAD:
            color = (230, 30, 30) if cf >= high else (255, 150, 0)
        else:
            color = (40, 180, 70)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{NAMES[c]} {cf:.2f}"
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0], tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=color)
        draw.text((x1 + 2, y1), label, fill=(255, 255, 255), font=font)
    if score >= high:
        verdict = "🔴 ALERT: send the site manager"
    elif score >= low:
        verdict = "🟠 HUMAN REVIEW: the guard checks this photo"
    else:
        verdict = "🟢 PASS: no action"
    text = (f"### {verdict}\nHighest 'head' (no hard hat) score in this photo: **{score:.2f}** "
            f"· review band [{low:.2f}, {high:.2f}). Red boxes: head ≥ high threshold; "
            f"orange: head inside the review band; green: helmet. Boxes below "
            f"{floor:.2f} are hidden.")
    return img, text


def on_upload(image, low, high):
    boxes = detect(image)
    img, text = render(image, boxes, low, high)
    return boxes, img, text


def on_threshold(low, high, image, boxes):
    return (*update_costs(low, high), *render(image, boxes, low, high))


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
assumptions_table = pd.DataFrame(
    {"assumption": list(A.keys()), "value": [str(v) for v in A.values()]})
per_class = read_csv("per_class_test.csv")
cm_boxes = read_csv("confusion_matrix_boxes_test.csv")
baselines = read_csv("baselines_test.csv")
splits = read_csv("dataset_splits.csv")
sizes = read_csv("dataset_object_sizes.csv")
ds = DECISION["dataset"]

with gr.Blocks(title="Hard-hat compliance: costed operating point") as demo:
    gr.Markdown(
        "# Hard-hat compliance on a construction site\n"
        f"A finetuned YOLO11n flags camera photos where someone is not wearing a hard hat "
        f"(positive class: `{DECISION['positive_class']}`). Dataset: Roboflow Universe "
        f"`{ds['workspace']}/{ds['project']}` v{ds['version']}. One case = one camera photo; "
        f"its score is the highest `head` confidence in it. Costs use the test split "
        f"reweighted to a production base rate of {A['BASE_RATE']:.0%}."
    )
    with gr.Row():
        low = gr.Slider(0, 1, value=LOW0, step=0.01, label="Low threshold (below → PASS)")
        high = gr.Slider(0, 1, value=HIGH0, step=0.01, label="High threshold (at or above → ALERT)")
        reset = gr.Button(f"Reset to chosen ({LOW0:.2f} / {HIGH0:.2f})")

    with gr.Tabs():
        with gr.Tab("Cost & human review"):
            kpis = gr.Markdown()
            with gr.Row():
                expected_tbl = gr.Dataframe(label="Expected outcomes per 1,000 photos (production base rate)")
                raw_tbl = gr.Dataframe(label="Outcomes on the test split (raw counts)")
            curve_plot = gr.Plot(label="Cost against threshold")
            alt_tbl = gr.Dataframe(label="Current design against the alternatives")
            with gr.Row():
                if image_or_none("two_threshold_map.png"):
                    gr.Image(image_or_none("two_threshold_map.png"),
                             label="Cost of every (low, high) pair, chosen pair marked")
                gr.Dataframe(assumptions_table, label="Cost assumptions (justified in REPORT.md)")

        with gr.Tab("Try an image"):
            boxes_state = gr.State([])
            with gr.Row():
                image_in = gr.Image(type="pil", label="Upload a site photo")
                image_out = gr.Image(type="pil", label="Predictions at the current thresholds")
            verdict = gr.Markdown("Upload a photo to see the model's predictions.")

        with gr.Tab("Per-class results"):
            gr.Markdown("Box-level results on the test split. Positive class: `head` "
                        "(a head without a hard hat). Confusion matrix rows are predictions, "
                        "columns the ground truth; `background` means no box.")
            if per_class is not None:
                gr.Dataframe(per_class, label="Per-class precision, recall, AP50, mAP50-95")
            with gr.Row():
                if cm_boxes is not None:
                    gr.Dataframe(cm_boxes.rename(columns={"Unnamed: 0": ""}),
                                 label="Confusion matrix (boxes, test split)")
                if image_or_none("confusion_matrix_boxes_test.png"):
                    gr.Image(image_or_none("confusion_matrix_boxes_test.png"),
                             label="Confusion matrix (Ultralytics)")
            if baselines is not None:
                gr.Dataframe(baselines, label="Finetuned model against the two baselines (test split)")
            if splits is not None:
                gr.Dataframe(splits, label="Dataset: images and instances per split")
            if sizes is not None:
                gr.Dataframe(sizes, label="Dataset: object side relative to image side")

        with gr.Tab("Failure examples"):
            gr.Markdown("Test photos the model gets wrong at the chosen single threshold. "
                        "Top row: model; bottom row: ground-truth labels.")
            for name, label in [("failures_false_negatives.png", "Missed (false negatives)"),
                                ("failures_false_positives.png", "False alarms (false positives)")]:
                if image_or_none(name):
                    gr.Image(image_or_none(name), label=label)

    cost_outputs = [kpis, expected_tbl, raw_tbl, curve_plot, alt_tbl]
    for slider in (low, high):
        slider.change(on_threshold, [low, high, image_in, boxes_state],
                      cost_outputs + [image_out, verdict])
    image_in.change(on_upload, [image_in, low, high], [boxes_state, image_out, verdict])
    reset.click(lambda: (LOW0, HIGH0), None, [low, high])
    demo.load(update_costs, [low, high], cost_outputs)

if __name__ == "__main__":
    demo.launch()
