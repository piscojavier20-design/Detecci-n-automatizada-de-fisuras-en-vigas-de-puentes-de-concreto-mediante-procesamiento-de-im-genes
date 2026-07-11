# fis_col.py

import os
import sys
import csv
import math
import time
import traceback
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

# -------------------------
# Optional GUI (PyQt6)
# -------------------------
GUI_AVAILABLE = True
try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QLabel, QLineEdit, QPushButton,
        QVBoxLayout, QHBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox,
        QTextEdit, QMessageBox, QCheckBox
    )
except Exception:
    GUI_AVAILABLE = False

# -------------------------
# Basic helpers
# -------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

SUMMARY_FIELDNAMES = [
    "image",
    "crack_id",
    "longitud_cm",
    "abertura_superior_cm",
    "abertura_inferior_cm",
    "abertura_intermedia_cm",
    "inclinacion",
    "criterio_de_falla",
]

DETAIL_FIELDNAMES = [
    "image","crack_idx","beam_length_mm","beam_height_mm","beam_length_px","beam_height_px",
    "local_height_px","local_u_pct","local_window_pct","mm_per_px_x","mm_per_px_y","mm_per_px_y_global",
    "length_px","length_px_centerline","length_px_pca","length_method","length_mm","length_mm_centerline",
    "length_mm_pca","w15_mm","w50_mm","w85_mm","angle_deg","inclination_class","zone_h","zone_v","criterion"
]

def list_images(folder: str) -> List[str]:
    imgs = []
    for e in IMG_EXTS:
        imgs += glob(os.path.join(folder, f"*{e}"))
        imgs += glob(os.path.join(folder, f"*{e.upper()}"))
    imgs = sorted(set(imgs))
    return imgs

def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def open_path(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", path], check=False)
        else:
            import subprocess
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        pass


def get_default_device():
    """
    Selecciona GPU si CUDA está disponible en el entorno actual; de lo contrario usa CPU.
    Esto evita fallas cuando el script se ejecuta accidentalmente desde un entorno sin CUDA.
    """
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# -------------------------
# Scale CSV (compatible con dect_fis_col.py)
# -------------------------
SCALE_CSV_NAME = "scale_lengths.csv"


def ensure_scale_csv(input_dir: str, log_fn=print) -> str:
    """
    Ensures a scale CSV exists in input_dir.
    Formato compatible con dect_fis_col.py:
      image, length_distx_mm, length_disty_mm

    Convención:
      - length_distx_mm = longitud real de la viga (mm)
      - length_disty_mm = altura real de la viga (mm)
    """
    p = os.path.join(input_dir, SCALE_CSV_NAME)
    imgs = [os.path.basename(x) for x in list_images(input_dir)]
    if os.path.exists(p):
        return p

    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "length_distx_mm", "length_disty_mm"])
        w.writeheader()
        for im in imgs:
            w.writerow({"image": im, "length_distx_mm": 1.0, "length_disty_mm": 1.0})

    log_fn(f"[Scale] ALERTA: No existía '{SCALE_CSV_NAME}'. Lo creé con 1.0 mm para longitud y altura de viga.")
    log_fn("[Scale] Debes editarlo con la longitud real y la altura real de la viga por imagen.")
    return p


def load_scale_lengths_map(scale_csv_path: str, log_fn=print) -> Dict[str, Tuple[float, float]]:
    scale_map: Dict[str, Tuple[float, float]] = {}
    if not scale_csv_path or not os.path.exists(scale_csv_path):
        log_fn(f"[Scale] Aviso: no existe CSV de escala: {scale_csv_path}")
        return scale_map
    try:
        with open(scale_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = (row.get("image") or row.get("file") or row.get("filename") or "").strip()
                if not img:
                    continue
                try:
                    length_mm = float(row.get("length_distx_mm", "1.0"))
                    height_mm = float(row.get("length_disty_mm", "1.0"))
                except Exception:
                    length_mm, height_mm = 1.0, 1.0
                scale_map[os.path.basename(img)] = (length_mm, height_mm)
        log_fn(f"[Scale] Longitudes/alturas leídas: {len(scale_map)} filas desde {scale_csv_path}")
    except Exception as e:
        log_fn(f"[Scale] Error leyendo CSV de escala: {e}")
    return scale_map

# -------------------------
# Polygon -> mask
# -------------------------
def mask_from_poly(poly_xy: np.ndarray, W: int, H: int) -> np.ndarray:
    m = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(m)
    dr.polygon([tuple(map(float, p)) for p in poly_xy], outline=1, fill=1)
    return np.array(m, dtype=np.uint8)

# -------------------------
# Skeleton + diameter path
# -------------------------
_NEI8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def _neighbors8(y, x, H, W):
    for dy, dx in _NEI8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx < W:
            yield ny, nx, (1.0 if (dy == 0 or dx == 0) else math.sqrt(2.0))

def _dijkstra_prev_on_skel(sk_bin: np.ndarray, src_yx: Tuple[int, int]):
    import heapq
    H, W = sk_bin.shape
    sy, sx = int(src_yx[0]), int(src_yx[1])
    dist = {(sy, sx): 0.0}
    prev = {(sy, sx): None}
    pq = [(0.0, sy, sx)]
    far_node = (sy, sx)
    far_dist = 0.0

    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[(y, x)] + 1e-9:
            continue
        if d > far_dist:
            far_dist = d
            far_node = (y, x)

        for ny, nx, w in _neighbors8(y, x, H, W):
            if not sk_bin[ny, nx]:
                continue
            nd = d + w
            if (ny, nx) not in dist or nd < dist[(ny, nx)]:
                dist[(ny, nx)] = nd
                prev[(ny, nx)] = (y, x)
                heapq.heappush(pq, (nd, ny, nx))

    return prev, far_node, float(far_dist)

def _reconstruct_path(prev: dict, end_node: Tuple[int, int]) -> np.ndarray:
    path = []
    cur = end_node
    while cur is not None:
        y, x = cur
        path.append((float(x), float(y)))
        cur = prev.get(cur, None)
    path.reverse()
    return np.array(path, dtype=float)

def skeletonize_mask(mask_bin: np.ndarray) -> Optional[np.ndarray]:
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask_bin.astype(bool)).astype(np.uint8)
    except Exception:
        return None

def skeleton_diameter_path(sk_bin: np.ndarray) -> Tuple[Optional[np.ndarray], float, Optional[Tuple[int,int]], Optional[Tuple[int,int]]]:
    """
    Returns:
      pts_xy: Nx2 (x,y) along diameter path (curvilinear centerline)
      L_px: curvilinear length in px (using 8-neighborhood weights)
      a_yx, b_yx: endpoints
    """
    H, W = sk_bin.shape
    ys, xs = np.where(sk_bin)
    if len(ys) == 0:
        return None, 0.0, None, None

    # endpoints: degree 1 pixels
    deg = {}
    for y, x in zip(ys, xs):
        d = 0
        for dy, dx in _NEI8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and sk_bin[ny, nx]:
                d += 1
        deg[(y, x)] = d
    endpoints = [p for p, d in deg.items() if d == 1]
    if not endpoints:
        endpoints = [(int(ys[0]), int(xs[0]))]

    # from one endpoint -> farthest a
    prev0, a, _ = _dijkstra_prev_on_skel(sk_bin, endpoints[0])
    # from a -> farthest b and prev
    prev, b, L_px = _dijkstra_prev_on_skel(sk_bin, a)
    pts = _reconstruct_path(prev, b)
    return pts, float(L_px), a, b

def pca_fallback_line(mask_bin: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 2:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    pc1 = VT[0]
    proj = X @ pc1
    tmin, tmax = proj.min(), proj.max()
    p1 = (mu[0] + tmin * pc1)
    p2 = (mu[0] + tmax * pc1)
    return np.array([[p1[0], p1[1]], [p2[0], p2[1]]], dtype=float)  # (x,y)

# -------------------------
# Widths at 3 stations (curvilinear)
# -------------------------

def orient_centerline_bottom_to_top(pts_xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Fuerza la orientación física del eje central para que:
      - pts[0]   = extremo inferior
      - pts[-1]  = extremo superior

    Convención de imagen:
      - y crece hacia abajo
      - por tanto, el punto inferior suele tener mayor y

    Esto garantiza que:
      - 0.15 -> ancho inferior
      - 0.50 -> ancho intermedio
      - 0.85 -> ancho superior
    """
    if pts_xy is None:
        return pts_xy
    pts = np.asarray(pts_xy, dtype=float)
    if len(pts) < 2:
        return pts
    y0 = float(pts[0, 1])
    y1 = float(pts[-1, 1])

    # Queremos inicio ABAJO (mayor y) y final ARRIBA (menor y)
    if y0 < y1:
        pts = pts[::-1].copy()
    elif abs(y0 - y1) <= 1e-9:
        # desempate estable para casos casi horizontales
        if float(pts[0, 0]) > float(pts[-1, 0]):
            pts = pts[::-1].copy()
    return pts


def widths_three_stations_curvilinear(poly_xy: np.ndarray, W: int, H: int,
                                      stations=(0.15, 0.50, 0.85), max_step=700):
    """
    Returns:
      widths_px: [w15, w50, w85]
      segs_px:   [(x1,y1,x2,y2) ...] width segments to draw
      centers_px:[(cx,cy) ...]
      pts_center_px: Nx2 polyline for plotting (x,y)
    """
    mask = mask_from_poly(poly_xy, W, H).astype(bool)
    if mask.sum() == 0:
        return [0.0]*3, [(0,0,0,0)]*3, [(0,0)]*3, None

    sk = skeletonize_mask(mask.astype(np.uint8))
    pts = None
    if sk is not None and sk.any():
        pts, _, _, _ = skeleton_diameter_path(sk > 0)

    if pts is None or len(pts) < 2:
        pts = pca_fallback_line(mask.astype(np.uint8))
    if pts is None or len(pts) < 2:
        return [0.0]*3, [(0,0,0,0)]*3, [(0,0)]*3, None

    # Convención física fija:
    #   0.15 -> inferior
    #   0.50 -> intermedio
    #   0.85 -> superior
    pts = orient_centerline_bottom_to_top(pts)

    # arclength
    dxy = np.diff(pts, axis=0)
    seglen = np.hypot(dxy[:, 0], dxy[:, 1])
    total = float(seglen.sum())
    if total < 1e-9:
        cx, cy = float(pts[0, 0]), float(pts[0, 1])
        return [0.0]*3, [(cx,cy,cx,cy)]*3, [(cx,cy)]*3, pts

    cum = np.concatenate([[0.0], np.cumsum(seglen)])

    def point_normal_at(tfrac: float):
        target = tfrac * total
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = max(0, min(j, len(pts) - 2))
        t0, t1 = cum[j], cum[j+1]
        alpha = 0.0 if (t1 - t0) < 1e-9 else (target - t0) / (t1 - t0)
        p = (1 - alpha) * pts[j] + alpha * pts[j+1]
        tang = pts[j+1] - pts[j]
        nrm = float(np.hypot(tang[0], tang[1]))
        if nrm < 1e-9:
            tang = np.array([1.0, 0.0])
        else:
            tang = tang / nrm
        normal = np.array([-tang[1], tang[0]])
        return (float(p[0]), float(p[1])), (float(normal[0]), float(normal[1]))

    Hh, Ww = mask.shape

    def inb(ix, iy):
        return 0 <= ix < Ww and 0 <= iy < Hh

    widths, segs, centers = [], [], []
    for tf in stations:
        (cx, cy), (nx, ny) = point_normal_at(tf)
        centers.append((cx, cy))

        # step +
        ppx, ppy = cx, cy
        for _ in range(max_step):
            ix, iy = int(round(ppx)), int(round(ppy))
            if (not inb(ix, iy)) or (not mask[iy, ix]):
                break
            ppx += nx
            ppy += ny
        x_pos, y_pos = ppx, ppy

        # step -
        pmx, pmy = cx, cy
        for _ in range(max_step):
            ix, iy = int(round(pmx)), int(round(pmy))
            if (not inb(ix, iy)) or (not mask[iy, ix]):
                break
            pmx -= nx
            pmy -= ny
        x_neg, y_neg = pmx, pmy

        w = float(math.hypot(x_pos - x_neg, y_pos - y_neg))
        widths.append(w)
        segs.append((float(x_neg), float(y_neg), float(x_pos), float(y_pos)))

    return widths, segs, centers, pts


def beam_scale_from_poly(poly_xy: np.ndarray, W: int, H: int,
                         beam_length_mm: float, beam_height_mm: float,
                         stations=(0.45, 0.50, 0.55)):
    """
    Calcula mm/px anisotrópico a partir de la máscara de VIGA:
      - mm_per_px_x = beam_length_mm / longitud del eje central de la viga en px
      - mm_per_px_y = beam_height_mm / menor altura medida en 45%, 50% y 55% del eje
    """
    widths_px, _, _, pts_center_px = widths_three_stations_curvilinear(poly_xy, W, H, stations=stations)

    beam_length_px = 0.0
    if pts_center_px is not None and len(pts_center_px) >= 2:
        dxy = np.diff(pts_center_px, axis=0)
        beam_length_px = float(np.hypot(dxy[:, 0], dxy[:, 1]).sum())

    valid_heights = [float(w) for w in widths_px if w and w > 0]
    beam_height_px = min(valid_heights) if valid_heights else 0.0

    mmx = float(beam_length_mm / beam_length_px) if (beam_length_mm > 0 and beam_length_px > 1e-9) else 1.0
    mmy = float(beam_height_mm / beam_height_px) if (beam_height_mm > 0 and beam_height_px > 1e-9) else 1.0

    return {
        "beam_length_px": float(beam_length_px),
        "beam_height_px": float(beam_height_px),
        "beam_length_mm": float(beam_length_mm),
        "beam_height_mm": float(beam_height_mm),
        "mm_per_px_x": float(mmx),
        "mm_per_px_y": float(mmy),
        "beam_pts_center_px": pts_center_px,
    }


def point_at_fraction_on_polyline(pts_xy: Optional[np.ndarray], frac: float) -> Optional[np.ndarray]:
    if pts_xy is None or len(pts_xy) < 2:
        return None
    frac = float(min(1.0, max(0.0, frac)))
    dxy = np.diff(pts_xy, axis=0)
    seglen = np.hypot(dxy[:, 0], dxy[:, 1])
    total = float(seglen.sum())
    if total < 1e-9:
        return np.asarray(pts_xy[0], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    target = frac * total
    j = int(np.searchsorted(cum, target, side="right") - 1)
    j = max(0, min(j, len(pts_xy) - 2))
    t0, t1 = cum[j], cum[j+1]
    alpha = 0.0 if (t1 - t0) < 1e-9 else (target - t0) / (t1 - t0)
    return np.asarray((1 - alpha) * pts_xy[j] + alpha * pts_xy[j+1], dtype=float)


def nearest_fraction_on_polyline(pts_xy: Optional[np.ndarray], point_xy: np.ndarray) -> float:
    if pts_xy is None or len(pts_xy) < 2:
        return 0.5
    dxy = np.diff(pts_xy, axis=0)
    seglen = np.hypot(dxy[:, 0], dxy[:, 1])
    total = float(seglen.sum())
    if total < 1e-9:
        return 0.5
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    p = np.asarray(point_xy, dtype=float)
    best_dist = float("inf")
    best_s = 0.5 * total
    for i in range(len(dxy)):
        a = pts_xy[i]
        b = pts_xy[i+1]
        ab = b - a
        lab2 = float(ab[0] * ab[0] + ab[1] * ab[1])
        if lab2 < 1e-12:
            continue
        t = float(np.dot(p - a, ab) / lab2)
        t = min(1.0, max(0.0, t))
        q = a + t * ab
        dist = float(np.hypot(*(p - q)))
        if dist < best_dist:
            best_dist = dist
            best_s = float(cum[i] + t * seglen[i])
    return float(best_s / total) if total > 1e-9 else 0.5


def measure_mask_widths_on_centerline(mask: np.ndarray,
                                      pts_xy: Optional[np.ndarray],
                                      stations: List[float],
                                      max_step: int = 700):
    if pts_xy is None or len(pts_xy) < 2:
        return [0.0 for _ in stations], [(0.0, 0.0, 0.0, 0.0) for _ in stations], [(0.0, 0.0) for _ in stations]

    dxy = np.diff(pts_xy, axis=0)
    seglen = np.hypot(dxy[:, 0], dxy[:, 1])
    total = float(seglen.sum())
    if total < 1e-9:
        return [0.0 for _ in stations], [(0.0, 0.0, 0.0, 0.0) for _ in stations], [(0.0, 0.0) for _ in stations]
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    Hh, Ww = mask.shape

    def inb(ix, iy):
        return 0 <= ix < Ww and 0 <= iy < Hh

    widths, segs, centers = [], [], []
    for tf in stations:
        tf = float(min(1.0, max(0.0, tf)))
        target = tf * total
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = max(0, min(j, len(pts_xy) - 2))
        t0, t1 = cum[j], cum[j+1]
        alpha = 0.0 if (t1 - t0) < 1e-9 else (target - t0) / (t1 - t0)
        p = (1 - alpha) * pts_xy[j] + alpha * pts_xy[j+1]
        tang = pts_xy[j+1] - pts_xy[j]
        nrm = float(np.hypot(tang[0], tang[1]))
        tang = np.array([1.0, 0.0]) if nrm < 1e-9 else tang / nrm
        nx, ny = -float(tang[1]), float(tang[0])
        cx, cy = float(p[0]), float(p[1])
        centers.append((cx, cy))

        ppx, ppy = cx, cy
        for _ in range(max_step):
            ix, iy = int(round(ppx)), int(round(ppy))
            if (not inb(ix, iy)) or (not mask[iy, ix]):
                break
            ppx += nx
            ppy += ny
        x_pos, y_pos = ppx, ppy

        pmx, pmy = cx, cy
        for _ in range(max_step):
            ix, iy = int(round(pmx)), int(round(pmy))
            if (not inb(ix, iy)) or (not mask[iy, ix]):
                break
            pmx -= nx
            pmy -= ny
        x_neg, y_neg = pmx, pmy

        widths.append(float(math.hypot(x_pos - x_neg, y_pos - y_neg)))
        segs.append((float(x_neg), float(y_neg), float(x_pos), float(y_pos)))
    return widths, segs, centers


def crack_local_beam_vertical_scale(beam_poly_xy: np.ndarray,
                                    crack_pts_center_px: Optional[np.ndarray],
                                    W: int,
                                    H: int,
                                    beam_height_mm: float,
                                    window_frac: float = 0.05,
                                    n_samples: int = 11):
    beam_mask = mask_from_poly(beam_poly_xy, W, H).astype(bool)
    _, _, _, beam_pts_center_px = widths_three_stations_curvilinear(beam_poly_xy, W, H, stations=(0.45, 0.50, 0.55))
    crack_mid = point_at_fraction_on_polyline(crack_pts_center_px, 0.50)
    if crack_mid is None:
        return {
            "local_u_frac": 0.5,
            "local_window_frac": float(window_frac),
            "local_height_px": 0.0,
            "mm_per_px_y_local": 1.0,
            "beam_pts_center_px": beam_pts_center_px,
        }

    u0 = nearest_fraction_on_polyline(beam_pts_center_px, crack_mid)
    u1 = max(0.0, u0 - float(window_frac))
    u2 = min(1.0, u0 + float(window_frac))
    stations = np.linspace(u1, u2, max(3, int(n_samples))).tolist()
    widths_px, _, _ = measure_mask_widths_on_centerline(beam_mask, beam_pts_center_px, stations=stations)
    valid = [float(w) for w in widths_px if w and w > 0]
    local_height_px = min(valid) if valid else 0.0
    local_mmy = float(beam_height_mm / local_height_px) if (beam_height_mm > 0 and local_height_px > 1e-9) else 1.0
    return {
        "local_u_frac": float(u0),
        "local_window_frac": float(window_frac),
        "local_height_px": float(local_height_px),
        "mm_per_px_y_local": float(local_mmy),
        "beam_pts_center_px": beam_pts_center_px,
    }


def segment_length_mm_with_anisotropic_scale(pts_center_px: Optional[np.ndarray], mmx: float, mmy: float) -> float:
    if pts_center_px is None or len(pts_center_px) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts_center_px) - 1):
        dx = float(pts_center_px[i + 1, 0] - pts_center_px[i, 0])
        dy = float(pts_center_px[i + 1, 1] - pts_center_px[i, 1])
        total += math.hypot(dx * mmx, dy * mmy)
    return float(total)


def width_segment_mm(seg_px: Tuple[float, float, float, float], mmx: float, mmy: float) -> float:
    x1, y1, x2, y2 = seg_px
    return float(math.hypot((x2 - x1) * mmx, (y2 - y1) * mmy))


def poly_major_axis_length_px(poly_xy: np.ndarray) -> float:
    """Longitud principal del polígono en px usando PCA."""
    try:
        pts = np.asarray(poly_xy, dtype=float)
        if pts is None or len(pts) < 2:
            return 0.0
        mu = pts.mean(axis=0, keepdims=True)
        X = pts - mu
        _, _, VT = np.linalg.svd(X, full_matrices=False)
        pc1 = VT[0]
        proj = X @ pc1
        return float(max(0.0, proj.max() - proj.min()))
    except Exception:
        return 0.0


def robust_crack_length_metrics(poly_xy: np.ndarray,
                                pts_center_px: Optional[np.ndarray],
                                mmx: float,
                                mmy: float):
    """Medición robusta de longitud para evitar subestimaciones severas.

    Retorna un dict con:
      - length_px_centerline
      - length_px_pca
      - length_px_used
      - length_mm_centerline
      - length_mm_pca
      - length_mm_used
      - length_method

    Regla:
      1) usar la longitud curvilínea del eje central si existe;
      2) comparar contra la longitud principal PCA del polígono;
      3) quedarse con la MAYOR de ambas para evitar casos donde el skeleton
         queda truncado y reporta longitudes absurdamente pequeñas.
    """
    center_px = 0.0
    center_mm = 0.0
    if pts_center_px is not None and len(pts_center_px) >= 2:
        dxy = np.diff(pts_center_px, axis=0)
        center_px = float(np.hypot(dxy[:, 0], dxy[:, 1]).sum())
        center_mm = segment_length_mm_with_anisotropic_scale(pts_center_px, mmx, mmy)

    pca_px = poly_major_axis_length_px(poly_xy)
    pca_mm = pca_px * float(max(1e-12, 0.5 * (mmx + mmy)))

    if center_mm >= pca_mm and center_mm > 0.0:
        used_mm = center_mm
        used_px = center_px
        method = "centerline"
    elif pca_mm > 0.0:
        used_mm = pca_mm
        used_px = pca_px
        method = "pca_major_axis"
    else:
        used_mm = 0.0
        used_px = 0.0
        method = "none"

    return {
        "length_px_centerline": float(center_px),
        "length_px_pca": float(pca_px),
        "length_px_used": float(used_px),
        "length_mm_centerline": float(center_mm),
        "length_mm_pca": float(pca_mm),
        "length_mm_used": float(used_mm),
        "length_method": method,
    }

# -------------------------
# Inclination class
# -------------------------
def inclination_class_from_endpoints(x1, y1, x2, y2):
    dx = (x2 - x1)
    dy = (y2 - y1)
    ang = math.degrees(math.atan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0
    a = abs(ang) % 180.0
    if a > 90.0:
        a = 180.0 - a
    if a <= 15.0:
        cls = "Horizontal"
    elif a >= 75.0:
        cls = "Vertical"
    else:
        cls = "Diagonal"
    return float(a), cls

def crack_endpoints_from_centerline(pts_xy: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1 = float(pts_xy[0, 0]), float(pts_xy[0, 1])
    x2, y2 = float(pts_xy[-1, 0]), float(pts_xy[-1, 1])
    return x1, y1, x2, y2

# -------------------------
# Failure criterion heuristic (no beam bbox required)
# -------------------------
def zone_vertical_from_centroid(cy: float, H: int) -> str:
    ny = cy / max(1.0, float(H))
    if ny < 1/3:
        return "Superior"
    elif ny < 2/3:
        return "Central"
    else:
        return "Inferior"


def zone_horizontal_from_centroid(cx: float, W: int) -> str:
    x = float(cx) / max(1.0, float(W))
    if x < 1/4:
        return "Izquierda"
    if x < 3/4:
        return "Centro"
    return "Derecha"


def crack_zone_in_beam(x: float, y: float, bx1: float, by1: float, bx2: float, by2: float):
    """Zona relativa dentro del bbox de viga.

    - zone_h: tercios estrictos (Izquierda / Centro / Derecha).
    - zone_v: solo Superior / Inferior respecto al eje horizontal medio de la viga (aprox. por bbox).
      (Si más adelante quieres usar el eje medio de la MÁSCARA de viga, aquí es donde se reemplaza).
    """
    w = max(1e-9, (bx2 - bx1))
    h = max(1e-9, (by2 - by1))
    nx = (x - bx1) / w

    # Tercios horizontales
    if nx < 1/3:
        zh = "Izquierda"
    elif nx < 2/3:
        zh = "Centro"
    else:
        zh = "Derecha"

    # Superior / Inferior respecto al eje medio de la viga
    y_mid = (by1 + by2) / 2.0
    zv = "Superior" if y < y_mid else "Inferior"

    return zv, zh

def poly_bbox(poly: np.ndarray):
    pts = np.asarray(poly, dtype=float)
    xs = pts[:, 0]; ys = pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

def bbox_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1 <= bx2) and (ax2 >= bx1) and (ay1 <= by2) and (ay2 >= by1)


def classify_failure(inclination_class: str, zone_h: str, zone_v: str) -> str:
    """Heurística de criterio de falla (tú definición final):

    - Vertical + Centro -> Flexión
    - Diagonal + (Centro o Costados) -> Cortante  (la diagonal en centro es válida si V es alto)
    - Horizontal + Centro -> Compresión
    - Resto -> No estructural
    """
    # Cortante: diagonal (en cualquier tercio)
    if inclination_class == "Diagonal":
        return "Falla a Cortante"

    # Flexión: vertical y tercio central
    if inclination_class == "Vertical" and zone_h == "Centro":
        return "Falla a Flexión"

    # Compresión: horizontal y tercio central
    if inclination_class == "Horizontal" and zone_h == "Centro":
        return "Falla a Compresión"

    return "Fisura No Estructural"
# -------------------------
# Plot in mm (blue length + width lines)
# -------------------------
def plot_crack_geometry_mm(title: str,
                           poly_xy_px: np.ndarray,
                           pts_center_px: Optional[np.ndarray],
                           wsegs_px: List[Tuple[float,float,float,float]],
                           mmx: float, mmy: float,
                           out_png: str):
    import matplotlib.pyplot as plt

    safe_mkdir(os.path.dirname(out_png))

    # polygon to mm
    px = [float(p[0]) * mmx for p in poly_xy_px]
    py = [float(p[1]) * mmy for p in poly_xy_px]

    plt.figure()
    plt.fill(px, py, alpha=0.20, label="Máscara fisura")

    # centerline in mm
    if pts_center_px is not None and len(pts_center_px) >= 2:
        xs = [float(p[0]) * mmx for p in pts_center_px]
        ys = [float(p[1]) * mmy for p in pts_center_px]
        plt.plot(xs, ys, "-", linewidth=2, color="blue", label="Eje central")
        plt.plot(xs, ys, "o", markersize=2, color="blue")

    # widths
    labels = ["Ancho 15%", "Ancho 50%", "Ancho 85%"]
    colors = ["orange", "magenta", "green"]
    for (seg, lab, col) in zip(wsegs_px, labels, colors):
        x1,y1,x2,y2 = seg
        plt.plot([x1*mmx, x2*mmx], [y1*mmy, y2*mmy], "-", linewidth=3, color=col, label=lab)

    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.gca().invert_yaxis()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# -------------------------
# Overlay image
# -------------------------
def draw_overlay(img_path: str,
                 fis_polys: List[np.ndarray],
                 apo_polys: List[np.ndarray],
                 out_path: str):
    img = Image.open(img_path).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    for poly in fis_polys:
        dr.polygon([tuple(map(float, p)) for p in poly], fill=(255,0,0,60), outline=(255,0,0,180))
    for poly in apo_polys:
        dr.polygon([tuple(map(float, p)) for p in poly], fill=(0,255,0,60), outline=(0,255,0,180))
    safe_mkdir(os.path.dirname(out_path))
    img.save(out_path)

# -------------------------
# Ultralytics wrapper
# -------------------------
def require_ultralytics():
    try:
        from ultralytics import YOLO  # noqa
        return True, ""
    except Exception as e:
        return False, str(e)


def detect_beam_boxes(det_model, img_path: str, conf: float = 0.25, device=None):
    """Detecta vigas y retorna bboxes [x1,y1,x2,y2] en píxeles.
    Si el modelo tiene máscaras, las cajas se derivan de dichas máscaras.
    Si no, usa r.boxes.xyxy.
    """
    if device is None:
        device = get_default_device()
    r = det_model.predict(source=img_path, conf=conf, device=device, verbose=False, save=False)[0]
    boxes = []

    # Prioridad: máscaras/polígonos (coherente con dect_fis_col.py si el modelo es seg)
    try:
        if getattr(r, "masks", None) is not None and getattr(r.masks, "xy", None) is not None:
            for poly in r.masks.xy:
                pp = np.asarray(poly, dtype=float)
                if pp is None or len(pp) < 3:
                    continue
                xs = pp[:, 0]; ys = pp[:, 1]
                boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    except Exception:
        pass

    # Fallback: cajas directas
    if not boxes:
        if getattr(r, "boxes", None) is None or r.boxes.xyxy is None:
            return boxes
        xyxy = r.boxes.xyxy.cpu().numpy()
        for b in xyxy:
            x1, y1, x2, y2 = map(float, b[:4])
            boxes.append([x1, y1, x2, y2])

    boxes.sort(key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]), reverse=True)
    return boxes


def draw_overlay_with_beams(img_path: str,
                            beam_polys: List[np.ndarray],
                            beam_boxes: List[List[float]],
                            beam_centerlines: Optional[List[Optional[np.ndarray]]],
                            fis_polys: List[np.ndarray],
                            apo_polys: List[np.ndarray],
                            crack_annotations: List[dict],
                            out_path: str):
    """Overlay detallado.

    - beam_polys: máscara/polígono de viga detectada por DET/SEG.
    - beam_boxes: bbox auxiliar de cada viga.
    - beam_centerlines: eje central de cada viga si se pudo calcular.
    - crack_annotations: lista por fisura con ids y segmentos de medición.
    """
    img = Image.open(img_path).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")

    # --- Vigas ---
    for i, poly in enumerate(beam_polys or []):
        try:
            pts = [tuple(map(float, p)) for p in poly]
            dr.polygon(pts, fill=(0, 128, 255, 45), outline=(0, 128, 255, 210))
        except Exception:
            pass

        if beam_centerlines and i < len(beam_centerlines):
            cpts = beam_centerlines[i]
            if cpts is not None and len(cpts) >= 2:
                for j in range(len(cpts) - 1):
                    x1, y1 = map(float, cpts[j])
                    x2, y2 = map(float, cpts[j + 1])
                    dr.line([x1, y1, x2, y2], fill=(0, 255, 255, 230), width=3)

        if i < len(beam_boxes):
            x1, y1, x2, y2 = map(float, beam_boxes[i])
            dr.rectangle([x1, y1, x2, y2], outline=(0, 128, 255, 220), width=3)
            dr.text((x1 + 6, y1 + 6), f"Viga {i+1}", fill=(255, 255, 255, 255))

    # --- Apoyos ---
    for poly in apo_polys:
        try:
            dr.polygon([tuple(map(float, p)) for p in poly], fill=(0,255,0,60), outline=(0,255,0,190))
        except Exception:
            pass

    # --- Fisuras + mediciones ---
    width_colors = [(255, 165, 0, 255), (255, 0, 255, 255), (0, 255, 0, 255)]
    width_tags = ["15%", "50%", "85%"]

    for ann in crack_annotations:
        poly = ann.get("poly")
        if poly is not None and len(poly) >= 3:
            try:
                dr.polygon([tuple(map(float, p)) for p in poly], fill=(255,0,0,55), outline=(255,0,0,190))
            except Exception:
                pass

        crack_id = ann.get("crack_id", "F?")
        center_pts = ann.get("centerline_pts")
        if center_pts is not None and len(center_pts) >= 2:
            for j in range(len(center_pts) - 1):
                x1, y1 = map(float, center_pts[j])
                x2, y2 = map(float, center_pts[j + 1])
                dr.line([x1, y1, x2, y2], fill=(255, 255, 0, 230), width=2)
            p_mid = point_at_fraction_on_polyline(np.asarray(center_pts, dtype=float), 0.50)
            if p_mid is not None:
                label = str(crack_id)
                if ann.get("length_mm") is not None and ann.get("w50_mm") is not None:
                    try:
                        label = f"{crack_id} | L={float(ann.get('length_mm', 0.0)):.2f} mm | w50={float(ann.get('w50_mm', 0.0)):.2f} mm"
                    except Exception:
                        label = str(crack_id)
                dr.text((float(p_mid[0]) + 6, float(p_mid[1]) + 6), label, fill=(255, 255, 0, 255))

        for k, seg in enumerate(ann.get("width_segments", [])[:3]):
            x1, y1, x2, y2 = map(float, seg)
            col = width_colors[k]
            tag = width_tags[k]
            dr.line([x1, y1, x2, y2], fill=col, width=3)
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            dr.text((mx + 4, my + 4), f"{crack_id}-{tag}", fill=col)

    safe_mkdir(os.path.dirname(out_path))
    img.save(out_path)

def segment_objects(seg_model, img_path: str, conf: float, device=None):
    """
    Returns dict with polys by class name matching.
      out = {"fisura":[poly...], "apoyo":[poly...], "viga":[poly...]}
    """
    if device is None:
        device = get_default_device()
    r = seg_model.predict(source=img_path, conf=conf, device=device, verbose=False, save=False)[0]
    out = {"fisura": [], "apoyo": [], "viga": []}
    if getattr(r, "masks", None) is None:
        return out

    cls = r.boxes.cls.cpu().numpy().astype(int)
    polys = r.masks.xy

    # id->name
    try:
        names = {int(i): str(n).lower() for i, n in r.names.items()}
    except Exception:
        names = {int(i): str(i) for i in set(cls)}

    for c, poly in zip(cls, polys):
        lab = names.get(int(c), "")
        if "fisur" in lab or "crack" in lab:
            out["fisura"].append(np.asarray(poly, dtype=float))
        elif "apoyo" in lab or "support" in lab:
            out["apoyo"].append(np.asarray(poly, dtype=float))
        elif "viga" in lab or "beam" in lab:
            out["viga"].append(np.asarray(poly, dtype=float))
        else:
            # unknown class -> ignore
            pass
    return out

# -------------------------
# Output rows
# -------------------------
@dataclass
class CrackDiag:
    image: str
    crack_idx: int
    length_mm: float
    w15_mm: float
    w50_mm: float
    w85_mm: float
    inclination_class: str
    criterion: str

def write_csv(path: str, rows: List[dict], fieldnames: Optional[List[str]] = None):
    safe_mkdir(os.path.dirname(path))
    if not fieldnames:
        fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

# -------------------------
# Core analysis
# -------------------------
def analyze_folder(input_dir: str,
                 seg_weights: str,
                 det_weights: str,
                 conf_seg: float = 0.25,
                 conf_det: float = 0.25,
                 make_overlays: bool = True,
                 make_plots: bool = True,
                 device=None,
                 log_fn=print) -> str:
    """
    Runs analysis on all images in input_dir.
    Returns output root path (input_dir).
    """
    ok, err = require_ultralytics()
    if not ok:
        raise RuntimeError(f"No se pudo importar ultralytics. Instala con: pip install ultralytics\nDetalle: {err}")

    from ultralytics import YOLO

    if device is None:
        device = get_default_device()
    log_fn(f"[Device] Ejecutando inferencia con device={device}")

    imgs = list_images(input_dir)
    if not imgs:
        raise RuntimeError("No encontré imágenes en la carpeta.")

    # scale csv
    scale_csv = ensure_scale_csv(input_dir, log_fn=log_fn)
    scale_map = load_scale_lengths_map(scale_csv, log_fn=log_fn)

    # output structure
    crack_results_dir = os.path.join(input_dir, "Crack_Results")
    results_dir       = os.path.join(input_dir, "Results")
    overlays_dir      = os.path.join(results_dir, "overlays")
    plots_dir         = os.path.join(results_dir, "plots_mm")
    safe_mkdir(crack_results_dir)
    safe_mkdir(results_dir)

    # load model
    log_fn(f"[Model] Cargando pesos SEG (segmentación): {seg_weights}")
    seg_model = YOLO(seg_weights)
    log_fn(f"[Model] Cargando pesos DET (detección): {det_weights}")
    det_model = YOLO(det_weights)

    # technical CSV rows
    seg_rows_all: List[dict] = []
    diag_rows_all: List[dict] = []
    diag_rows_summary_all: List[dict] = []

    t0 = time.time()
    for idx_img, ip in enumerate(imgs, start=1):
        img_path = ip
        bn = os.path.basename(ip)
        try:
            im = Image.open(ip).convert("RGB")
            W, H = im.size
        except Exception as e:
            log_fn(f"[WARN] No pude abrir la imagen: {bn} | {e}")
            continue

        try:
            beam_boxes = detect_beam_boxes(det_model, img_path, conf=conf_det, device=device)
        except Exception as e:
            log_fn(f"[WARN] Falló la detección en: {bn} | {e}")
            continue

        beam_length_mm, beam_height_mm = scale_map.get(bn, (1.0, 1.0))
        beam_length_mm = float(beam_length_mm)
        beam_height_mm = float(beam_height_mm)

        seg = segment_objects(seg_model, ip, conf=conf_seg, device=device)
        fis_polys = seg["fisura"]
        apo_polys = seg["apoyo"]

        det_seg = segment_objects(det_model, ip, conf=conf_det, device=device)
        beam_polys = det_seg.get("viga", [])

        mmx, mmy = 1.0, 1.0
        beam_scale_info = {
            "beam_length_px": 0.0,
            "beam_height_px": 0.0,
            "beam_length_mm": beam_length_mm,
            "beam_height_mm": beam_height_mm,
            "mm_per_px_x": 1.0,
            "mm_per_px_y": 1.0,
            "beam_pts_center_px": None,
        }
        if beam_polys:
            try:
                chosen_beam_poly = max(beam_polys, key=lambda p: float((np.max(p[:,0]) - np.min(p[:,0])) * (np.max(p[:,1]) - np.min(p[:,1]))))
                beam_scale_info = beam_scale_from_poly(chosen_beam_poly, W, H, beam_length_mm, beam_height_mm)
                mmx = float(beam_scale_info["mm_per_px_x"])
                mmy = float(beam_scale_info["mm_per_px_y"])
            except Exception:
                pass

        log_fn(
            f"[{idx_img}/{len(imgs)}] {bn} -> fisuras={len(fis_polys)} apoyos={len(apo_polys)} vigas={len(beam_polys)} | "
            f"beam_px=({beam_scale_info['beam_length_px']:.3f},{beam_scale_info['beam_height_px']:.3f}) | "
            f"scale=({mmx:.8f},{mmy:.8f})"
        )

        # Preparación de geometría de vigas detectadas con DET/SEG
        beam_boxes = []
        beam_centerlines: List[Optional[np.ndarray]] = []
        for bp in beam_polys:
            try:
                beam_boxes.append(list(poly_bbox(bp)))
                _, _, _, pts_center_bp = widths_three_stations_curvilinear(bp, W, H)
                beam_centerlines.append(pts_center_bp)
            except Exception:
                beam_boxes.append(list(poly_bbox(bp)))
                beam_centerlines.append(None)
        if (not beam_boxes):
            try:
                beam_boxes = detect_beam_boxes(det_model, img_path, conf=conf_det, device=device)
            except Exception:
                beam_boxes = []

        crack_annotations: List[dict] = []

        # per crack
        for cidx, poly in enumerate(fis_polys, start=1):
            if poly is None or len(poly) < 3:
                continue

            widths_px, wsegs_px, centers_px, pts_center_px = widths_three_stations_curvilinear(poly, W, H)

            if pts_center_px is not None and len(pts_center_px) >= 2:
                x1, y1, x2, y2 = crack_endpoints_from_centerline(pts_center_px)
            else:
                xs = poly[:,0]; ys = poly[:,1]
                x1, y1 = float(xs.min()), float(ys.min())
                x2, y2 = float(xs.max()), float(ys.max())

            ang_deg, incl_cls = inclination_class_from_endpoints(x1, y1, x2, y2)
            cx = float(np.mean(poly[:, 0]))
            cy = float(np.mean(poly[:, 1]))
            zone_h = zone_horizontal_from_centroid(cx, W)
            zone_v = zone_vertical_from_centroid(cy, H)

            crack_local_scale = {
                "local_u_frac": 0.5,
                "local_window_frac": 0.05,
                "local_height_px": beam_scale_info.get("beam_height_px", 0.0),
                "mm_per_px_y_local": float(mmy),
            }
            try:
                crack_bb = poly_bbox(poly)
                chosen = None
                chosen_poly = None
                best_area = 0.0
                for bb in (beam_boxes or []):
                    if not bbox_overlap(crack_bb, bb):
                        continue
                    ix1 = max(crack_bb[0], bb[0]); iy1 = max(crack_bb[1], bb[1])
                    ix2 = min(crack_bb[2], bb[2]); iy2 = min(crack_bb[3], bb[3])
                    area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    if area > best_area:
                        best_area = area
                        chosen = bb
                if chosen is not None:
                    zone_v, zone_h = crack_zone_in_beam(cx, cy, *chosen)
                if beam_polys:
                    for bp in beam_polys:
                        try:
                            bbp = poly_bbox(bp)
                            if not bbox_overlap(crack_bb, bbp):
                                continue
                            ix1 = max(crack_bb[0], bbp[0]); iy1 = max(crack_bb[1], bbp[1])
                            ix2 = min(crack_bb[2], bbp[2]); iy2 = min(crack_bb[3], bbp[3])
                            area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                            if area >= best_area:
                                best_area = area
                                chosen_poly = bp
                        except Exception:
                            continue
                if chosen_poly is None and beam_polys:
                    chosen_poly = max(beam_polys, key=lambda p: float((np.max(p[:,0]) - np.min(p[:,0])) * (np.max(p[:,1]) - np.min(p[:,1]))))
                if chosen_poly is not None:
                    crack_local_scale = crack_local_beam_vertical_scale(
                        chosen_poly,
                        pts_center_px,
                        W,
                        H,
                        beam_height_mm,
                        window_frac=0.05,
                        n_samples=11,
                    )
            except Exception:
                pass

            local_mmy = float(crack_local_scale.get("mm_per_px_y_local", mmy) or mmy)
            w15_mm = width_segment_mm(wsegs_px[0], mmx, local_mmy)
            w50_mm = width_segment_mm(wsegs_px[1], mmx, local_mmy)
            w85_mm = width_segment_mm(wsegs_px[2], mmx, local_mmy)

            length_metrics = robust_crack_length_metrics(poly, pts_center_px, mmx, local_mmy)
            length_mm = float(length_metrics["length_mm_used"])
            length_px = float(length_metrics["length_px_used"])
            if length_mm <= 0.0:
                dx = x2 - x1
                dy = y2 - y1
                length_px = float(math.hypot(dx, dy))
                length_mm = math.hypot(dx * mmx, dy * local_mmy)
                length_metrics["length_px_used"] = float(length_px)
                length_metrics["length_mm_used"] = float(length_mm)
                length_metrics["length_method"] = "endpoints_fallback"

            criterion = classify_failure(incl_cls, zone_h, zone_v)

            diag = CrackDiag(
                image=bn,
                crack_idx=cidx,
                length_mm=length_mm,
                w15_mm=w15_mm,
                w50_mm=w50_mm,
                w85_mm=w85_mm,
                inclination_class=incl_cls,
                criterion=criterion
            )
            detail_row = {
                "image": diag.image,
                "crack_idx": diag.crack_idx,
                "beam_length_mm": round(beam_scale_info["beam_length_mm"], 4),
                "beam_height_mm": round(beam_scale_info["beam_height_mm"], 4),
                "beam_length_px": round(beam_scale_info["beam_length_px"], 4),
                "beam_height_px": round(beam_scale_info["beam_height_px"], 4),
                "local_height_px": round(float(crack_local_scale.get("local_height_px", 0.0)), 4),
                "local_u_pct": round(100.0 * float(crack_local_scale.get("local_u_frac", 0.5)), 4),
                "local_window_pct": round(100.0 * float(crack_local_scale.get("local_window_frac", 0.05)), 4),
                "mm_per_px_x": round(mmx, 8),
                "mm_per_px_y": round(local_mmy, 8),
                "mm_per_px_y_global": round(mmy, 8),
                "length_px": round(length_metrics["length_px_used"], 4),
                "length_px_centerline": round(length_metrics["length_px_centerline"], 4),
                "length_px_pca": round(length_metrics["length_px_pca"], 4),
                "length_method": length_metrics["length_method"],
                "length_mm": round(diag.length_mm, 4),
                "length_mm_centerline": round(length_metrics["length_mm_centerline"], 4),
                "length_mm_pca": round(length_metrics["length_mm_pca"], 4),
                "w15_mm": round(diag.w15_mm, 4),
                "w50_mm": round(diag.w50_mm, 4),
                "w85_mm": round(diag.w85_mm, 4),
                "angle_deg": round(ang_deg, 4),
                "inclination_class": diag.inclination_class,
                "zone_h": zone_h,
                "zone_v": zone_v,
                "criterion": diag.criterion,
            }
            diag_rows_all.append(detail_row)

            summary_row = {
                "image": diag.image,
                "crack_id": f"F{diag.crack_idx}",
                "longitud_cm": round(diag.length_mm / 10.0, 2),
                "abertura_superior_cm": round(diag.w15_mm / 10.0, 2),
                "abertura_inferior_cm": round(diag.w85_mm / 10.0, 2),
                "abertura_intermedia_cm": round(diag.w50_mm / 10.0, 2),
                "inclinacion": diag.inclination_class,
                "criterio_de_falla": diag.criterion,
            }
            diag_rows_summary_all.append(summary_row)

            per_path = os.path.join(crack_results_dir, f"{Path(bn).stem}_crack{cidx}_details.csv")
            write_csv(per_path, [detail_row], fieldnames=DETAIL_FIELDNAMES)

            # technical segments CSV (for traceability / future)
            if pts_center_px is not None and len(pts_center_px) >= 2:
                for s in range(len(pts_center_px) - 1):
                    xA,yA = pts_center_px[s]
                    xB,yB = pts_center_px[s+1]
                    seg_rows_all.append({
                        "image": bn,
                        "crack_idx": cidx,
                        "seg_idx": s+1,
                        "x1_px": float(xA), "y1_px": float(yA),
                        "x2_px": float(xB), "y2_px": float(yB),
                        "dx_px": float(xB-xA),
                        "dy_px": float(yB-yA),
                        "mm_per_px_x": mmx,
                        "mm_per_px_y": local_mmy,
                        "mm_per_px_y_global": mmy,
                        "local_height_px": float(crack_local_scale.get("local_height_px", 0.0)),
                        "local_u_pct": 100.0 * float(crack_local_scale.get("local_u_frac", 0.5)),
                        "local_window_pct": 100.0 * float(crack_local_scale.get("local_window_frac", 0.05)),
                        "inclination_deg": round(ang_deg, 4),
                        "inclination_class": incl_cls,
                        "zone_v": zone_v,
                        "criterion": criterion,
                        "w15_mm": round(w15_mm, 4),
                        "w50_mm": round(w50_mm, 4),
                        "w85_mm": round(w85_mm, 4),
                        # width segs in px (to replicate plots later)
                        "w15_x1_px": wsegs_px[0][0], "w15_y1_px": wsegs_px[0][1], "w15_x2_px": wsegs_px[0][2], "w15_y2_px": wsegs_px[0][3],
                        "w50_x1_px": wsegs_px[1][0], "w50_y1_px": wsegs_px[1][1], "w50_x2_px": wsegs_px[1][2], "w50_y2_px": wsegs_px[1][3],
                        "w85_x1_px": wsegs_px[2][0], "w85_y1_px": wsegs_px[2][1], "w85_x2_px": wsegs_px[2][2], "w85_y2_px": wsegs_px[2][3],
                    })

            crack_annotations.append({
                "crack_idx": cidx,
                "crack_id": f"F{cidx}",
                "poly": poly,
                "centerline_pts": pts_center_px,
                "width_segments": wsegs_px,
                "beam_idx": None,
                "criterion": criterion,
                "length_mm": length_mm,
                "w50_mm": w50_mm,
            })

            # plot mm
            if make_plots:
                try:
                    out_png = os.path.join(plots_dir, f"{Path(bn).stem}_crack{cidx}.png")
                    plot_crack_geometry_mm(
                        title=f"{bn} – crack {cidx} | {incl_cls} | {criterion}",
                        poly_xy_px=poly,
                        pts_center_px=pts_center_px,
                        wsegs_px=wsegs_px,
                        mmx=mmx, mmy=local_mmy,
                        out_png=out_png
                    )
                except Exception:
                    pass

        if make_overlays:
            try:
                out_ov = os.path.join(overlays_dir, f"{Path(bn).stem}.jpg")
                draw_overlay_with_beams(
                    ip,
                    beam_polys=beam_polys,
                    beam_boxes=beam_boxes,
                    beam_centerlines=beam_centerlines,
                    fis_polys=fis_polys,
                    apo_polys=apo_polys,
                    crack_annotations=crack_annotations,
                    out_path=out_ov,
                )
            except Exception:
                pass

    # write consolidated outputs
    diag_all_csv = os.path.join(crack_results_dir, "crack_diagnosis_all.csv")
    write_csv(diag_all_csv, diag_rows_summary_all, fieldnames=SUMMARY_FIELDNAMES)

    diag_details_csv = os.path.join(crack_results_dir, "crack_diagnosis_details_all.csv")
    write_csv(diag_details_csv, diag_rows_all, fieldnames=DETAIL_FIELDNAMES)

    seg_csv = os.path.join(results_dir, "crack_segments.csv")
    if seg_rows_all:
        write_csv(seg_csv, seg_rows_all)
    else:
        write_csv(seg_csv, [], fieldnames=["image","crack_idx","seg_idx","x1_px","y1_px","x2_px","y2_px"])

    log_fn(f"\n[OK] Listo. Diagnóstico consolidado: {diag_all_csv}")
    log_fn(f"[OK] Segmentos técnicos: {seg_csv}")
    log_fn(f"[OK] Overlays: {overlays_dir if make_overlays else '(omitido)'}")
    log_fn(f"[OK] Plots mm: {plots_dir if make_plots else '(omitido)'}")
    log_fn(f"[Time] {time.time()-t0:.1f} s")

    return input_dir

# -------------------------
# GUI Thread Worker
# -------------------------
if GUI_AVAILABLE:
    class Worker(QThread):
        log = pyqtSignal(str)
        done = pyqtSignal(str)
        failed = pyqtSignal(str)

        def __init__(self, input_dir, seg_weights, det_weights, conf_seg, conf_det, make_overlays, make_plots):
            super().__init__()
            self.input_dir = input_dir
            self.seg_weights = seg_weights
            self.det_weights = det_weights
            self.conf_seg = conf_seg
            self.conf_det = conf_det
            self.make_overlays = make_overlays
            self.make_plots = make_plots

        def run(self):
            try:
                out = analyze_folder(
                    input_dir=self.input_dir,
                    seg_weights=self.seg_weights,
                    det_weights=self.det_weights,
                    conf_seg=self.conf_seg,
                    conf_det=self.conf_det,
                    make_overlays=self.make_overlays,
                    make_plots=self.make_plots,
                    device=get_default_device(),
                    log_fn=lambda s: self.log.emit(str(s))
                )
                self.done.emit(out)
            except Exception as e:
                msg = f"{e}\n\n{traceback.format_exc()}"
                self.failed.emit(msg)

    class App(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("fis_col.py — Crack Analyzer (SEG + mm + diagnóstico)")
            self.worker = None

            # Inputs
            self.inp_dir = QLineEdit()
            self.btn_browse_dir = QPushButton("Elegir carpeta de imágenes")
            self.btn_browse_dir.clicked.connect(self.pick_dir)

            self.seg_weights = QLineEdit()
            self.btn_browse_w = QPushButton("Elegir pesos SEG (best.pt)")
            self.btn_browse_w.clicked.connect(self.pick_weights)

            self.det_weights = QLineEdit()
            self.btn_browse_det = QPushButton("Elegir pesos DET (best.pt)")
            self.btn_browse_det.clicked.connect(self.pick_det_weights)

            self.conf_det = QDoubleSpinBox()
            self.conf_det.setRange(0.01, 0.99)
            self.conf_det.setSingleStep(0.05)
            self.conf_det.setValue(0.25)

            self.conf = QDoubleSpinBox()
            self.conf.setRange(0.01, 0.99)
            self.conf.setSingleStep(0.05)
            self.conf.setValue(0.25)

            self.chk_overlays = QCheckBox("Guardar overlays (detecciones)")
            self.chk_overlays.setChecked(True)

            self.chk_plots = QCheckBox("Guardar plots_mm (longitud azul + anchos)")
            self.chk_plots.setChecked(True)

            self.btn_run = QPushButton("RUN")
            self.btn_run.clicked.connect(self.on_run)

            self.log = QTextEdit()
            self.log.setReadOnly(True)

            # Layout
            L = QVBoxLayout()

            row1 = QHBoxLayout()
            row1.addWidget(QLabel("Carpeta:"))
            row1.addWidget(self.inp_dir)
            row1.addWidget(self.btn_browse_dir)
            L.addLayout(row1)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel("Pesos SEG:"))
            row2.addWidget(self.seg_weights)
            row2.addWidget(self.btn_browse_w)
            L.addLayout(row2)

            row2b = QHBoxLayout()
            row2b.addWidget(QLabel("Pesos DET:"))
            row2b.addWidget(self.det_weights)
            row2b.addWidget(self.btn_browse_det)
            L.addLayout(row2b)

            row3 = QHBoxLayout()
            row3.addWidget(QLabel("conf SEG:"))
            row3.addWidget(self.conf)
            row3.addWidget(QLabel("conf DET:"))
            row3.addWidget(self.conf_det)
            row3.addStretch(1)
            L.addLayout(row3)

            row4 = QHBoxLayout()
            row4.addWidget(self.chk_overlays)
            row4.addWidget(self.chk_plots)
            row4.addStretch(1)
            L.addLayout(row4)

            L.addWidget(self.btn_run)
            L.addWidget(QLabel("Log:"))
            L.addWidget(self.log)

            self.setLayout(L)

        def pick_dir(self):
            d = QFileDialog.getExistingDirectory(self, "Selecciona carpeta")
            if d:
                self.inp_dir.setText(d)

        def pick_weights(self):
            f, _ = QFileDialog.getOpenFileName(self, "Selecciona best.pt", filter="PyTorch Weights (*.pt);;All (*.*)")
            if f:
                self.seg_weights.setText(f)

        def append_log(self, s: str):
            self.log.append(s)

        
        def pick_det_weights(self):
            f, _ = QFileDialog.getOpenFileName(self, "Selecciona best.pt (DET)", filter="PyTorch Weights (*.pt);;All (*.*)")
            if f:
                self.det_weights.setText(f)

        def on_run(self):
            inp = self.inp_dir.text().strip()
            w   = self.seg_weights.text().strip()
            if not inp or not os.path.isdir(inp):
                QMessageBox.critical(self, "Error", "Selecciona una carpeta válida con imágenes.")
                return
            if not w or not os.path.exists(w):
                QMessageBox.critical(self, "Error", "Selecciona un archivo de pesos SEG (.pt).")
                return

            self.btn_run.setEnabled(False)
            self.log.clear()
            self.append_log("Iniciando...\n")

            d = self.det_weights.text().strip()
            if not d or not os.path.exists(d):
                QMessageBox.critical(self, "Error", "Selecciona un archivo de pesos DET (.pt).")
                self.btn_run.setEnabled(True)
                return

            self.worker = Worker(
                input_dir=inp,
                seg_weights=w,
                det_weights=d,
                conf_seg=float(self.conf.value()),
                conf_det=float(self.conf_det.value()),
                make_overlays=self.chk_overlays.isChecked(),
                make_plots=self.chk_plots.isChecked()
            )
            self.worker.log.connect(self.append_log)
            self.worker.done.connect(self.on_done)
            self.worker.failed.connect(self.on_failed)
            self.worker.start()

        def on_done(self, outdir: str):
            self.btn_run.setEnabled(True)
            self.append_log("\n✅ Terminado.")
            # open output folder
            try:
                open_path(outdir)
            except Exception:
                pass
            QMessageBox.information(
                self, "OK",
                "Listo.\n\nRevisar:\n- Crack_Results/crack_diagnosis_all.csv\n- Results/ (overlays, plots_mm, crack_segments.csv)\n\nSi scale_lengths.csv quedó en 1.0, se debe editar con la longitud y altura real de la viga y volver a correr."
            )

        def on_failed(self, msg: str):
            self.btn_run.setEnabled(True)
            self.append_log("\n❌ Error:\n" + msg)
            QMessageBox.critical(self, "Error", msg)

# -------------------------
# CLI fallback
# -------------------------
def cli_main():
    import argparse
    ap = argparse.ArgumentParser("fis_col.py — CLI")
    ap.add_argument("--input", required=True, help="Carpeta con imágenes")
    ap.add_argument("--seg_weights", required=True, help="best.pt de segmentación (YOLO-seg)")
    ap.add_argument("--det_weights", required=True, help="best.pt de detección (YOLO-detect) para ubicar la VIGA")
    ap.add_argument("--conf_seg", type=float, default=0.25, help="Confianza para segmentación")
    ap.add_argument("--conf_det", type=float, default=0.25, help="Confianza para detección")
    ap.add_argument("--device", default=None, help='cpu o índice GPU (por ejemplo 0). Si se omite, detecta automáticamente.')
    ap.add_argument("--no_overlays", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    analyze_folder(

        input_dir=args.input,
        seg_weights=args.seg_weights,
        det_weights=args.det_weights,
        conf_det=args.conf_det,
        conf_seg=args.conf_seg,
        make_overlays=(not args.no_overlays),
        make_plots=(not args.no_plots),
        device=args.device,
        log_fn=print
    )

if __name__ == "__main__":
    if GUI_AVAILABLE:
        app = QApplication(sys.argv)
        w = App()
        w.resize(1050, 650)
        w.show()
        sys.exit(app.exec())
    else:
        print("[WARN] PyQt6 no disponible. Ejecutando en modo CLI.")
        cli_main()