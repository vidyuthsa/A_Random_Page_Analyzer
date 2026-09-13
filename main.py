"""
Handwriting Analysis Toolkit — all four gag projects, unifiable in one run.

Modes:
    doodle_index      -> Neurotic Doodling Index (scribble density -> boredom/neurosis)
    signature_shake   -> Signature Shake Detector (two signatures -> hand stability)
    tremor            -> Drawing Tremor Monitor (straight line(s) -> hand tremor)
    doodle_pattern    -> Doodling Pattern Analyzer (doodle spread -> spatial heat map)
    auto              -> Looks at the photo, figures out what's actually in it
                         (line drawing? signature pair? general handwriting page?)
                         and runs + combines whichever analyses apply.

Usage:
    python main.py --mode auto samples/any_photo.jpg
    python main.py --mode doodle_index samples/notebook.jpg
    python main.py --mode signature_shake samples/two_signatures.jpg
    python main.py --mode tremor samples/line.jpg
    python main.py --mode doodle_pattern samples/doodles.jpg

Requires: opencv-python, numpy, scipy
    pip install opencv-python numpy scipy
"""

import argparse
import os
import random

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_and_threshold(path, thresh_val=150):
    """Load image, return (original_bgr, grayscale, binary_ink_mask)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return img, gray, thresh


def connected_components(thresh, min_size=50):
    """Return list of dicts: centroid, bbox, size, pixel coords for each ink cluster."""
    labeled, num = ndimage.label(thresh)
    clusters = []
    for i in range(1, num + 1):
        pixels = np.argwhere(labeled == i)
        if len(pixels) < min_size:
            continue
        y, x = pixels[:, 0], pixels[:, 1]
        clusters.append({
            "pixels": pixels,
            "centroid": (int(x.mean()), int(y.mean())),
            "bbox": (int(x.min()), int(y.min()), int(x.max()), int(y.max())),
            "size": len(pixels),
        })
    return clusters


def save_result(img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"Saved annotated image -> {out_path}")


def put_lines(img, lines, start_y=40, gap=35, color=(0, 0, 255), scale=0.8):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (20, start_y + i * gap),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _pca_line_fit(pixels):
    """Fit a line to (y, x) pixel coords using PCA (handles any angle,
    including near-vertical lines that break plain y=f(x) regression)."""
    pts = pixels[:, ::-1].astype(np.float64)  # -> (x, y) order
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    direction = eigvecs[:, 0]            # principal axis (the "ideal" line)
    normal = eigvecs[:, 1]               # perpendicular axis (deviation axis)
    along = centered @ direction         # position along the line
    perp = centered @ normal             # perpendicular distance = wobble
    elongation = eigvals[0] / (eigvals[1] + 1e-6)
    return mean, direction, along, perp, elongation


# ---------------------------------------------------------------------------
# CONTENT CLASSIFIER — decides what kind of photo this is
# ---------------------------------------------------------------------------

def classify_content(thresh, h, w):
    """
    Returns one of: "tremor", "signature", "handwriting"

    Heuristics:
    - Mostly a few long, thin, elongated strokes covering most of the ink
      and few total clusters -> someone drew straight line(s) -> tremor test.
    - A small number (2-6) of compact, non-full-width blobs, each looking
      like a signature (not a whole page of writing) -> signature pair.
    - Otherwise -> general handwriting / doodle page.
    """
    clusters = connected_components(thresh, min_size=100)
    if not clusters:
        return "handwriting"  # nothing detected; fall back safely

    total_pixels = sum(c["size"] for c in clusters)

    elongated_pixels = 0
    for c in clusters:
        _, _, _, _, elongation = _pca_line_fit(c["pixels"])
        if elongation > 6:
            elongated_pixels += c["size"]

    if total_pixels > 0 and elongated_pixels / total_pixels > 0.55 and len(clusters) <= 20:
        return "tremor"

    large_blobs = [c for c in clusters if c["size"] > 0.01 * h * w]
    if 2 <= len(large_blobs) <= 6 and len(clusters) < 40:
        widths = [c["bbox"][2] - c["bbox"][0] for c in large_blobs]
        if all(wd < w * 0.7 for wd in widths):
            return "signature"

    return "handwriting"


# ---------------------------------------------------------------------------
# 1. NEUROTIC DOODLING INDEX  (scribble-density based)
# ---------------------------------------------------------------------------

def analyze_doodle_index(gray, thresh):
    """Returns (annotated_result_img, [metric_lines])."""
    h, w = thresh.shape
    page_area = h * w
    ink_mask = (thresh > 0).astype(np.float32)

    # Local ink DENSITY, not presence. Normal handwriting (even messy
    # cursive) covers maybe 15-25% of a local window. Heavy cross-outs,
    # tight scribble loops, and scratched-out words push that density
    # way higher because ink is doubling/tripling back over itself.
    win = max(31, (min(h, w) // 20) | 1)
    density = cv2.boxFilter(ink_mask, ddepth=-1, ksize=(win, win))

    baseline = density[density > 0.02].mean() if np.any(density > 0.02) else 0.05
    hot_thresh = baseline * 1.8
    hot_mask = (density > hot_thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel)
    hot_clusters = connected_components(hot_mask, min_size=int(page_area * 0.0008))

    hot_area = sum(c["size"] for c in hot_clusters)
    scribble_fraction = hot_area / page_area

    boredom_level = min(99, scribble_fraction * 400)
    neurosis = min(99, boredom_level * random.uniform(0.9, 1.1))
    attention_span = max(0, 100 - boredom_level - len(hot_clusters) * 1.5)

    if attention_span < 20:
        attention_label = "COLLAPSING"
    elif attention_span < 45:
        attention_label = "CRITICAL"
    elif attention_span < 70:
        attention_label = "MODERATE"
    else:
        attention_label = "FINE"

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    heat_color = cv2.applyColorMap(
        (np.clip(density / (hot_thresh * 1.5), 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET)
    hot_bool = hot_mask > 0
    result[hot_bool] = cv2.addWeighted(result, 0.4, heat_color, 0.6, 0)[hot_bool]

    for c in hot_clusters:
        x0, y0, x1, y1 = c["bbox"]
        cv2.rectangle(result, (x0, y0), (x1, y1), (0, 0, 255), 2)

    lines = [
        f"BOREDOM LEVEL: {boredom_level:.0f}%",
        f"NEUROSIS INDEX: {neurosis:.0f}%",
        f"ATTENTION SPAN: {attention_label}",
        f"Scribble hotspots: {len(hot_clusters)}",
    ]
    return result, lines


def run_doodle_index(path, out_dir="output"):
    img, gray, thresh = load_and_threshold(path)
    result, lines = analyze_doodle_index(gray, thresh)
    put_lines(result, lines)
    save_result(result, os.path.join(out_dir, "doodle_index_result.jpg"))


# ---------------------------------------------------------------------------
# 2. SIGNATURE SHAKE DETECTOR
# ---------------------------------------------------------------------------

def analyze_signature_shake(gray, thresh):
    """Returns (overlay_img, [metric_lines]). Raises ValueError if not applicable."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 300]

    if len(bboxes) < 2:
        raise ValueError("Couldn't find two separate signature regions.")

    h, w = thresh.shape
    top_boxes = [b for b in bboxes if b[1] + b[3] / 2 < h / 2]
    bottom_boxes = [b for b in bboxes if b[1] + b[3] / 2 >= h / 2]
    if not top_boxes or not bottom_boxes:
        bboxes_sorted = sorted(bboxes, key=lambda b: b[1])
        mid = len(bboxes_sorted) // 2
        top_boxes, bottom_boxes = bboxes_sorted[:mid] or bboxes_sorted[:1], bboxes_sorted[mid:]

    def union_box(boxes):
        xs0 = min(b[0] for b in boxes)
        ys0 = min(b[1] for b in boxes)
        xs1 = max(b[0] + b[2] for b in boxes)
        ys1 = max(b[1] + b[3] for b in boxes)
        return xs0, ys0, xs1, ys1

    x0a, y0a, x1a, y1a = union_box(top_boxes)
    x0b, y0b, x1b, y1b = union_box(bottom_boxes)

    sig1 = thresh[y0a:y1a, x0a:x1a]
    sig2 = thresh[y0b:y1b, x0b:x1b]
    sig2_resized = cv2.resize(sig2, (sig1.shape[1], sig1.shape[0]))

    diff = cv2.absdiff(sig1, sig2_resized)
    matching = np.count_nonzero(diff == 0)
    match_pct = (matching / sig1.size) * 100

    stability = max(0, 100 - (100 - match_pct) * 1.5)
    caffeine = round(random.uniform(0.5, 4.5), 1)

    overlay = np.zeros((*sig1.shape, 3), dtype=np.uint8)
    overlay[:, :, 2] = sig1
    overlay[:, :, 1] = sig2_resized

    lines = [
        f"MATCH: {match_pct:.0f}%",
        f"STABILITY: {stability:.0f}%",
        f"CAFFEINE EST: {caffeine} cups",
    ]
    return overlay, lines


def run_signature_shake(path, out_dir="output"):
    img, gray, thresh = load_and_threshold(path)
    overlay, lines = analyze_signature_shake(gray, thresh)
    put_lines(overlay, lines, color=(255, 255, 255))
    save_result(overlay, os.path.join(out_dir, "signature_shake_result.jpg"))


# ---------------------------------------------------------------------------
# 3. DRAWING TREMOR MONITOR
# ---------------------------------------------------------------------------

def analyze_tremor(gray, thresh):
    """Returns (result_img, [metric_lines]). Raises ValueError if not applicable."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clusters = connected_components(closed, min_size=200)

    lines_found = []
    for c in clusters:
        mean, direction, along, perp, elongation = _pca_line_fit(c["pixels"])
        if elongation > 4:
            lines_found.append({"mean": mean, "direction": direction,
                                 "along": along, "perp": perp,
                                 "pixels": c["pixels"]})

    if not lines_found:
        raise ValueError("No line-like strokes detected.")

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    per_line_tremor = []

    for ln in lines_found:
        deviations = np.abs(ln["perp"])
        tremor_level = float(np.mean(deviations))
        per_line_tremor.append(tremor_level)

        for py, px in ln["pixels"]:
            cv2.circle(result, (int(px), int(py)), 1, (255, 0, 0), -1)

        a_min, a_max = ln["along"].min(), ln["along"].max()
        for t in np.arange(a_min, a_max, 6):
            pt = ln["mean"] + ln["direction"] * t
            cv2.circle(result, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

        dev_thresh = tremor_level * 1.5
        for (py, px), dev in zip(ln["pixels"], deviations):
            if dev > dev_thresh:
                cv2.circle(result, (int(px), int(py)), 2, (0, 0, 255), -1)

    tremor_level = float(np.mean(per_line_tremor))
    hand_control = max(0, 100 - tremor_level * 8)
    caffeine = round((tremor_level / 3) + random.uniform(0, 1.5), 1)

    metric_lines = [
        f"TREMOR: {tremor_level:.2f}px (avg of {len(lines_found)} lines)",
        f"HAND CONTROL: {hand_control:.0f}%",
        f"CAFFEINE EST: {caffeine} cups",
    ]
    return result, metric_lines


def run_tremor(path, out_dir="output"):
    img, gray, thresh = load_and_threshold(path)
    result, lines = analyze_tremor(gray, thresh)
    put_lines(result, lines)
    save_result(result, os.path.join(out_dir, "tremor_result.jpg"))


# ---------------------------------------------------------------------------
# 4. DOODLING PATTERN ANALYZER
# ---------------------------------------------------------------------------

def analyze_doodle_pattern(gray, thresh):
    """Returns (result_img, [metric_lines]). Raises ValueError if not applicable."""
    h, w = thresh.shape
    clusters = connected_components(thresh, min_size=50)

    if not clusters:
        raise ValueError("No doodle clusters detected.")

    centroids = np.array([c["centroid"] for c in clusters])

    quad_counts = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    for cx, cy in centroids:
        key = ("T" if cy < h / 2 else "B") + ("L" if cx < w / 2 else "R")
        quad_counts[key] += 1

    dominant_quad = max(quad_counts, key=quad_counts.get)
    concentration = (quad_counts[dominant_quad] / len(clusters)) * 100
    chaos_level = (np.std(centroids, axis=0).sum() / (w + h)) * 100
    boredom_level = min(99, concentration + random.uniform(10, 30))

    heat_map = np.zeros((h, w), dtype=np.float32)
    for cx, cy in centroids:
        cv2.circle(heat_map, (int(cx), int(cy)), 50, 1, -1)
    heat_map = cv2.GaussianBlur(heat_map, (51, 51), 0)
    heat_map_color = cv2.applyColorMap((heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(result, 0.6, heat_map_color, 0.4, 0)

    for c in clusters:
        x0, y0, x1, y1 = c["bbox"]
        cv2.rectangle(result, (x0, y0), (x1, y1), (0, 255, 0), 2)
    for cx, cy in centroids:
        cv2.circle(result, (int(cx), int(cy)), 3, (255, 0, 0), -1)

    lines = [
        f"BOREDOM: {boredom_level:.0f}%",
        f"CHAOS LEVEL: {chaos_level:.0f}%",
        f"HOTSPOT: {dominant_quad} ({quad_counts[dominant_quad]} clusters)",
        f"TOTAL DOODLES: {len(clusters)}",
    ]
    return result, lines


def run_doodle_pattern(path, out_dir="output"):
    img, gray, thresh = load_and_threshold(path)
    result, lines = analyze_doodle_pattern(gray, thresh)
    put_lines(result, lines)
    save_result(result, os.path.join(out_dir, "doodle_pattern_result.jpg"))


# ---------------------------------------------------------------------------
# AUTO MODE — one photo in, combined result depending on content
# ---------------------------------------------------------------------------

def _hstack_with_banner(images, banner_lines, banner_height=170):
    """Stack images side by side (equal height) under one shared metrics banner."""
    target_h = min(im.shape[0] for im in images)
    resized = []
    for im in images:
        scale = target_h / im.shape[0]
        new_w = int(im.shape[1] * scale)
        resized.append(cv2.resize(im, (new_w, target_h)))
    divider = np.full((target_h, 4, 3), (80, 80, 80), dtype=np.uint8)
    strip = resized[0]
    for im in resized[1:]:
        strip = np.hstack([strip, divider, im])

    banner = np.zeros((banner_height, strip.shape[1], 3), dtype=np.uint8)
    put_lines(banner, banner_lines, start_y=35, gap=32, color=(0, 0, 255), scale=0.75)
    return np.vstack([banner, strip])


def analyze_auto(gray, thresh):
    h, w = thresh.shape
    content_type = classify_content(thresh, h, w)
    
    combined_lines = []

    if content_type == "tremor":
        try:
            result, lines = analyze_tremor(gray, thresh)
            combined_lines = [f"CONTENT DETECTED: Line drawing / tremor test"] + lines
        except ValueError:
            # fall back if the line-detector was wrong about elongation
            result, lines = analyze_doodle_index(gray, thresh)
            content_type = "handwriting"
            combined_lines = ["CONTENT DETECTED: Handwriting page (tremor fallback)"] + lines
        final = _hstack_with_banner([result], combined_lines)

    elif content_type == "signature":
        try:
            result, lines = analyze_signature_shake(gray, thresh)
            combined_lines = ["CONTENT DETECTED: Signature pair"] + lines
        except ValueError:
            result, lines = analyze_doodle_index(gray, thresh)
            content_type = "handwriting"
            combined_lines = ["CONTENT DETECTED: Handwriting page (signature pair not found)"] + lines
        final = _hstack_with_banner([result], combined_lines)

    else:  # handwriting / doodle page -> combine BOTH scribble-density + spatial pattern
        result_a, lines_a = analyze_doodle_index(gray, thresh)
        result_b, lines_b = analyze_doodle_pattern(gray, thresh)
        put_lines(result_a, ["-- SCRIBBLE DENSITY --"], start_y=result_a.shape[0] - 20,
                  color=(255, 255, 255), scale=0.6)
        put_lines(result_b, ["-- SPATIAL PATTERN --"], start_y=result_b.shape[0] - 20,
                  color=(255, 255, 255), scale=0.6)
        combined_lines = ["CONTENT DETECTED: Handwriting / doodle page"] + lines_a + lines_b
        final = _hstack_with_banner([result_a, result_b], combined_lines, banner_height=260)

    return final, combined_lines

def run_auto(path, out_dir="output"):
    img, gray, thresh = load_and_threshold(path)
    final, _ = analyze_auto(gray, thresh)
    save_result(final, os.path.join(out_dir, "auto_combined_result.jpg"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODES = {
    "doodle_index": run_doodle_index,
    "signature_shake": run_signature_shake,
    "tremor": run_tremor,
    "doodle_pattern": run_doodle_pattern,
    "auto": run_auto,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake-metric handwriting analysis toolkit")
    parser.add_argument("image", help="Path to input photo")
    parser.add_argument("--mode", required=True, choices=MODES.keys())
    parser.add_argument("--out", default="output", help="Output directory")
    args = parser.parse_args()

    MODES[args.mode](args.image, args.out)
