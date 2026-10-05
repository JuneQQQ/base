"""DICOM study -> fixed-size uint8 slice stack.

Lessons from the forum that this module encodes (each one cost people 0.01-0.03 AUC):
  * File names are SOP Instance UIDs: sorting them gives a random slice order. Sort by
    ImagePositionPatient projected on the slice normal (InstanceNumber only as fallback).
  * Crop a fixed physical field of view (mm, from PixelSpacing) before resizing, so a knee
    occupies the same fraction of the frame whatever the acquisition matrix/FOV.
  * Don't keep only the central 20-80% of a stack: collaterals and the lateral meniscus
    live in the outer sagittal slices.
  * Mixed transfer syntaxes (JPEG Lossless / JPEG 2000) need pylibjpeg or gdcm installed.
  * Two training files have truncated PixelData; never let one bad file kill a study.
"""
from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
import pydicom

from .config import DEFAULT_PREP, PrepConfig

_HDR_TAGS = [
    "ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber",
    "PixelSpacing", "Rows", "Columns", "Laterality", "ImageLaterality",
]
# expected (row axis, column axis) and column sign per plane, in DICOM LPS coordinates:
# axial: rows run +x (to patient left), columns +y (to posterior)
# coronal: rows +x, columns -z (head at top); sagittal: rows +y, columns -z
_CANON = {"axial": (0, 1, +1), "coronal": (0, 2, -1), "sagittal": (1, 2, -1)}
_PLANE_OF_NORMAL_AXIS = {0: "sagittal", 1: "coronal", 2: "axial"}


def read_header(path: str) -> dict | None:
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=_HDR_TAGS)
    except Exception:
        return None
    h = {"path": path}
    iop = getattr(d, "ImageOrientationPatient", None)
    ipp = getattr(d, "ImagePositionPatient", None)
    h["iop"] = np.asarray(iop, np.float64) if iop is not None and len(iop) == 6 else None
    h["ipp"] = np.asarray(ipp, np.float64) if ipp is not None and len(ipp) == 3 else None
    try:
        h["inst"] = int(getattr(d, "InstanceNumber", 0) or 0)
    except Exception:
        h["inst"] = 0
    ps = getattr(d, "PixelSpacing", None)
    h["ps"] = (float(ps[0]), float(ps[1])) if ps is not None and len(ps) == 2 else None
    h["rows"] = int(getattr(d, "Rows", 0) or 0)
    h["cols"] = int(getattr(d, "Columns", 0) or 0)
    lat = str(getattr(d, "Laterality", "") or getattr(d, "ImageLaterality", "") or "").upper()
    h["lat"] = lat if lat in ("L", "R") else None
    return h


def series_geometry(headers: list[dict]) -> dict:
    """Plane (from the slice normal), slice order and knee side for one series."""
    ref = next((h for h in headers if h["iop"] is not None), None)
    if ref is None:
        order = sorted(range(len(headers)), key=lambda i: (headers[i]["inst"], headers[i]["path"]))
        return {"plane": None, "order": order, "side": _vote_side(headers, None), "iop": None}
    row, col = ref["iop"][:3], ref["iop"][3:]
    normal = np.cross(row, col)
    n_ax = int(np.argmax(np.abs(normal)))
    sign = 1.0 if normal[n_ax] >= 0 else -1.0

    def key(i):
        h = headers[i]
        if h["ipp"] is None:
            return (1, h["inst"], h["path"])
        return (0, float(np.dot(h["ipp"], normal)) * sign, h["path"])

    # ascending position along +x (sagittal), +y (coronal) or +z (axial)
    order = sorted(range(len(headers)), key=key)
    return {"plane": _PLANE_OF_NORMAL_AXIS[n_ax], "order": order,
            "side": _vote_side(headers, ref), "iop": ref["iop"]}


def _vote_side(headers: list[dict], ref: dict | None) -> str | None:
    """'L' / 'R': explicit Laterality tag first, else the image centre in patient space
    (LPS: x < 0 is the patient's right). Forum check: geometry agrees with the tag ~98%."""
    tags = [h["lat"] for h in headers if h["lat"]]
    if tags:
        return max(set(tags), key=tags.count)
    if ref is None or ref["ipp"] is None or ref["ps"] is None:
        return None
    row, col = ref["iop"][:3], ref["iop"][3:]
    centre = (ref["ipp"] + row * (ref["cols"] - 1) / 2 * ref["ps"][1]
              + col * (ref["rows"] - 1) / 2 * ref["ps"][0])
    return "R" if centre[0] < 0 else "L"


def decode(path: str) -> np.ndarray:
    d = pydicom.dcmread(path)
    a = d.pixel_array
    if a.ndim == 3:  # multi-frame / RGB oddities: keep the first plane
        a = a[0] if a.shape[0] < a.shape[-1] else a[..., 0]
    a = a.astype(np.float32)
    slope = float(getattr(d, "RescaleSlope", 1) or 1)
    inter = float(getattr(d, "RescaleIntercept", 0) or 0)
    a = a * slope + inter
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def canonicalize(img: np.ndarray, iop: np.ndarray | None, plane: str | None) -> np.ndarray:
    """Flip/transpose so every scanner's image of a given plane has the same orientation."""
    if iop is None or plane not in _CANON:
        return img
    row, col = iop[:3], iop[3:]
    want_r, want_c, want_c_sign = _CANON[plane]
    r_ax, c_ax = int(np.argmax(np.abs(row))), int(np.argmax(np.abs(col)))
    if (r_ax, c_ax) == (want_c, want_r):  # rotated 90 degrees: swap image axes
        img = img.T
        row, col = col, row
        r_ax, c_ax = c_ax, r_ax
    if r_ax == want_r and row[r_ax] < 0:
        img = img[:, ::-1]
    if c_ax == want_c and np.sign(col[c_ax]) != want_c_sign:
        img = img[::-1, :]
    return np.ascontiguousarray(img)


def crop_resize_mm(img: np.ndarray, ps: tuple | None, crop_mm: float, size: int) -> np.ndarray:
    """Centre-crop a crop_mm x crop_mm square (zero-padding when the FOV is smaller)."""
    ps_r, ps_c = ps if ps else (0.5, 0.5)
    h, w = img.shape
    ch = int(round(crop_mm / max(ps_r, 1e-3)))
    cw = int(round(crop_mm / max(ps_c, 1e-3)))
    out = np.zeros((ch, cw), np.float32)
    sy, sx = max(0, (h - ch) // 2), max(0, (w - cw) // 2)
    dy, dx = max(0, (ch - h) // 2), max(0, (cw - w) // 2)
    hh, ww = min(h, ch), min(w, cw)
    out[dy:dy + hh, dx:dx + ww] = img[sy:sy + hh, sx:sx + ww]
    interp = cv2.INTER_AREA if max(ch, cw) > size else cv2.INTER_LINEAR
    return cv2.resize(out, (size, size), interpolation=interp)


def pick_series(rows: list[dict], plane: str, fluid: int, used: set, n_files: dict) -> dict | None:
    """Pick the series for one slot: right plane, preferred fluid flag, typical slice count."""
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if not cands:
        return None
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        cands = pref or cands
    # 2D clinical series have ~20-45 slices; prefer those over thin 3D stacks of 150+
    return min(cands, key=lambda r: abs(n_files.get(r["SeriesInstanceUID"], 0) - 32))


def sample_positions(n: int, k: int, span: tuple) -> np.ndarray:
    if n <= 1:
        return np.zeros(k, int)
    lo, hi = int(n * span[0]), max(int(n * span[0]), int(np.ceil(n * span[1])) - 1)
    return np.linspace(lo, min(hi, n - 1), k).round().astype(int)


def load_series(series_dir: str, k: int, prep: PrepConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (k, H, W) uint8 slices, (k,) validity mask and geometry info for one series."""
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")]
    headers = [h for h in (read_header(f) for f in files) if h is not None]
    out = np.zeros((k, prep.img_size, prep.img_size), np.uint8)
    valid = np.zeros(k, np.uint8)
    if not headers:
        return out, valid, {"plane": None, "side": None, "n": 0}
    geo = series_geometry(headers)
    ordered = [headers[i] for i in geo["order"]]
    picks = sample_positions(len(ordered), k, prep.span)
    imgs = []
    for p in picks:
        try:
            imgs.append(decode(ordered[p]["path"]))
        except Exception:
            imgs.append(None)
    good = [a for a in imgs if a is not None]
    if good:
        sample = np.concatenate([a.ravel()[::7] for a in good])
        lo, hi = np.percentile(sample, [prep.pct_lo, prep.pct_hi])
    else:
        lo, hi = 0.0, 1.0
    for j, (p, a) in enumerate(zip(picks, imgs)):
        if a is None:
            continue
        h = ordered[p]
        a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        if prep.canonical_orientation:
            a = canonicalize(a, h["iop"] if h["iop"] is not None else geo["iop"], geo["plane"])
        a = crop_resize_mm(a, h["ps"], prep.crop_mm, prep.img_size)
        out[j] = (a * 255).astype(np.uint8)
        valid[j] = 1
    return out, valid, {"plane": geo["plane"], "side": geo["side"], "n": len(ordered)}


def build_study(study_dir: str, series_rows: Iterable[dict], prep: PrepConfig = DEFAULT_PREP):
    """Build the fixed slot stack for one study.

    series_rows: rows of train_series.csv / test_series.csv for this study (dicts with
    SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive).
    Returns vol (S, H, W) uint8, mask (S,) uint8, meta dict.
    """
    rows = list(series_rows)
    n_files = {}
    for r in rows:
        d = os.path.join(study_dir, str(r["SeriesInstanceUID"]))
        n_files[r["SeriesInstanceUID"]] = len(os.listdir(d)) if os.path.isdir(d) else 0
    S = prep.n_slices
    vol = np.zeros((S, prep.img_size, prep.img_size), np.uint8)
    mask = np.zeros(S, np.uint8)
    used, sides, slot_meta = set(), [], []
    for (plane, fluid, k), (a, b) in zip(prep.slots, prep.slot_bounds()):
        r = pick_series(rows, plane, fluid, used, n_files)
        if r is None or n_files.get(r["SeriesInstanceUID"], 0) == 0:
            slot_meta.append(None)
            continue
        used.add(r["SeriesInstanceUID"])
        try:
            sl, ok, info = load_series(os.path.join(study_dir, str(r["SeriesInstanceUID"])), k, prep)
        except Exception:
            slot_meta.append(None)
            continue
        vol[a:b], mask[a:b] = sl, ok
        if info["side"]:
            sides.append(info["side"])
        slot_meta.append({"series": r["SeriesInstanceUID"], **info})
    side = max(set(sides), key=sides.count) if sides else None
    if prep.normalize_laterality and side == "R":
        # mirror the right knee into a left knee: x is the row axis for axial/coronal
        # images and the slice axis for sagittal stacks
        for m, (a, b) in zip(slot_meta, prep.slot_bounds()):
            if m is None:
                continue
            if m["plane"] == "sagittal":
                vol[a:b], mask[a:b] = vol[a:b][::-1].copy(), mask[a:b][::-1].copy()
            elif m["plane"] in ("axial", "coronal"):
                vol[a:b] = vol[a:b][:, :, ::-1].copy()
    return vol, mask, {"side": side, "slots": slot_meta}
