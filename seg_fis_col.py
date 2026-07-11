# seg_fis_col.py — End-to-end (SEGMENTACIÓN Fisura/Apoyo)
# Flujo: audit -> split -> YAML -> train(seg) -> val/test (vis) -> clasificación Viga Fisurada/No Fisurada
# -> CSVs de longitudes (val/test/inferencia) -> tidy -> prune

import os, re, shutil, random, argparse, sys, csv, subprocess, math, heapq
from glob import glob
from typing import List, Tuple, Dict
from math import sqrt
import numpy as np
from PIL import Image, ImageDraw

# =========================
# Utilidades básicas
# =========================

import json
from pathlib import Path

def load_split_manifest(split_from: str):
    
    base = Path(split_from)

    js = base / "split_manifest.json"
    if js.exists():
        data = json.loads(js.read_text(encoding="utf-8"))
        return data["train"], data["val"], data["test"]

    tr = (base / "split_train.txt").read_text(encoding="utf-8").splitlines()
    va = (base / "split_val.txt").read_text(encoding="utf-8").splitlines()
    te = (base / "split_test.txt").read_text(encoding="utf-8").splitlines()
    return tr, va, te

def normalize_img_exts(img_dir):
    renames = 0
    if not os.path.isdir(img_dir): return renames
    for f in os.listdir(img_dir):
        src = os.path.join(img_dir, f)
        if not os.path.isfile(src): continue
        name, ext = os.path.splitext(f)
        if ext and ext != ext.lower():
            dst = os.path.join(img_dir, name + ext.lower())
            if not os.path.exists(dst):
                os.rename(src, dst); renames += 1
    return renames

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}[_-]", re.IGNORECASE)
NUMID_RE = re.compile(r"^\d{5,}[_-]")

def strip_prefix(stem):
    s = UUID_RE.sub("", stem); s = NUMID_RE.sub("", s); return s

def image_stems(img_dir):
    stems = set()
    if not os.path.isdir(img_dir): return stems
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
        for p in glob(os.path.join(img_dir, ext)):
            stems.add(os.path.splitext(os.path.basename(p))[0])
    return stems

def label_stems(lbl_dir):
    if not os.path.isdir(lbl_dir): return set()
    return {os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(lbl_dir, "*.txt"))}

def get_image_path_by_stem(stem, img_dir):
    for ext in (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"):
        p = os.path.join(img_dir, stem+ext)
        if os.path.exists(p): return p
    return None

# =========================
# Abrir / mostrar archivos
# =========================
def open_path(path):
    if not os.path.exists(path): return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return True
    except Exception:
        return False

def open_folder(folder_path, label="Carpeta"):
    ok = open_path(folder_path)
    print((f"{label} abierto: " if ok else f"{label} en: ") + folder_path)
    return ok

# =========================
# Auditoría y negativos
# =========================
def audit_and_fix_labels(img_dir, lbl_dir):
    imgs = image_stems(img_dir)
    os.makedirs(lbl_dir, exist_ok=True)
    labels = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]

    fixed = 0
    collisions = 0
    nomatch = []

    # 1) intento: match exacto o quitando prefijos UUID/NUMID
    for lf in labels:
        src = os.path.join(lbl_dir, lf)
        stem = os.path.splitext(lf)[0]

        if stem in imgs:
            continue

        cand = strip_prefix(stem)
        if cand in imgs:
            dst = os.path.join(lbl_dir, cand + ".txt")
            if os.path.exists(dst):
                collisions += 1
            else:
                os.rename(src, dst)
                fixed += 1
        else:
            nomatch.append(lf)

    # 2) intento extra: quitar primer token antes de _ o -
    remaining = []
    for lf in nomatch:
        stem = os.path.splitext(lf)[0]

        split_try = re.sub(r"^[^_-]+[_-]", "", stem)

        candidates = {
            s for s in imgs
            if (s == stem or s == split_try or s in stem or stem in s or s.endswith(split_try))
        }

        if len(candidates) == 1:
            c = list(candidates)[0]
            src = os.path.join(lbl_dir, lf)
            dst = os.path.join(lbl_dir, c + ".txt")
            if not os.path.exists(dst):
                os.rename(src, dst)
                fixed += 1
            else:
                collisions += 1
        else:
            remaining.append(lf)

    imgs2 = image_stems(img_dir)
    labs2 = label_stems(lbl_dir)
    pairs = len(imgs2 & labs2)

    print("==== AUDITORÍA ====")
    print(f"Imágenes totales: {len(imgs2)}")
    print(f"Labels totales:   {len(labs2)}")
    print(f"Pares válidos:    {pairs}")
    print(f"Labels renombrados: {fixed}")
    print(f"Conflictos: {collisions}")
    if remaining:
        print(f"Labels aún sin match: {len(remaining)} (ej.: {remaining[:10]})")

    return pairs

def create_empty_labels_from_missing(img_dir, lbl_dir, max_count=None, seed=42):
    imgs = image_stems(img_dir); labs = label_stems(lbl_dir)
    missing = sorted(list(imgs - labs))
    if not missing: return 0
    random.seed(seed); random.shuffle(missing)
    take = len(missing) if max_count is None else min(max_count, len(missing))
    created = 0
    for stem in missing[:take]:
        out = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(out):
            with open(out, "w", encoding="utf-8") as f: f.write("")
            created += 1
    if created: print(f"[Negativos] Labels vacíos creados: {created}  (de {len(missing)} posibles)")
    return created

# =========================
# Split y YAML (sobre seg_ds)
# =========================
def collect_pairs(img_dir, lbl_dir): 
    return sorted(image_stems(img_dir) & label_stems(lbl_dir))

def ensure_split_dirs(base):
    for p in ["images/train","images/val","images/test","labels/train","labels/val","labels/test",
              "labels_det/train","labels_det/val","labels_det/test"]:
        os.makedirs(os.path.join(base, p), exist_ok=True)

def clean_split_dirs(base):
    for d in [os.path.join(base, "images", s) for s in ["train","val","test"]] + \
             [os.path.join(base, "labels", s) for s in ["train","val","test"]] + \
             [os.path.join(base, "labels_det", s) for s in ["train","val","test"]]:
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp): os.remove(fp)

def copy_pair_seg(stem, split, img_dir, seg_lbl_dir, base):
    src_img = get_image_path_by_stem(stem, img_dir)
    if src_img:
        shutil.copy(src_img, os.path.join(base, "images", split, os.path.basename(src_img)))
    lsrc = os.path.join(seg_lbl_dir, stem+".txt")
    shutil.copy(lsrc, os.path.join(base, "labels", split, stem+".txt"))

def copy_det_label_if_exists(stem, split, det_lbl_src, base):
    p = os.path.join(det_lbl_src, stem+".txt")
    if os.path.exists(p):
        shutil.copy(p, os.path.join(base, "labels_det", split, stem+".txt"))

def smart_split(n, p_train=0.8, p_val=0.1, p_test=0.1, *, seed=None):
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        # aleatorio: todo a train, o a val/test si quieres; aquí dejamos train
        return (1, 0, 0)
    if n == 2:
        # reparte aleatorio entre (1,1,0) o (1,0,1)
        rng = random.Random(seed)
        return (1, 1, 0) if rng.random() < 0.5 else (1, 0, 1)

    # n >= 3: garantiza al menos 1 por split y el resto aleatorio
    rng = random.Random(seed)
    a = rng.randint(1, n - 2)         # deja sitio para v y s
    b = rng.randint(a + 1, n - 1)     # asegura que s >= 1
    t = a
    v = b - a
    s = n - b
    return (t, v, s)

def write_yaml(base, names):
    yaml_path = os.path.join(base, "seg_data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: " + base.replace('\\','/') + "\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for i,n in enumerate(names): f.write(f"  {i}: {n}\n")
    return yaml_path

def verify_split_integrity(base, strict=True):
    def stems_in(folder):
        return {os.path.splitext(f)[0] for f in os.listdir(folder)}
    img_train = stems_in(os.path.join(base, "images", "train"))
    img_val   = stems_in(os.path.join(base, "images", "val"))
    img_test  = stems_in(os.path.join(base, "images", "test"))
    lab_train = stems_in(os.path.join(base, "labels", "train"))
    lab_val   = stems_in(os.path.join(base, "labels", "val"))
    lab_test  = stems_in(os.path.join(base, "labels", "test"))

    overlap_iv = (img_train & img_val) | (img_train & img_test) | (img_val & img_test)
    overlap_lv = (lab_train & lab_val) | (lab_train & lab_test) | (lab_val & lab_test)

    miss_t = (img_train - lab_train) | (lab_train - img_train)
    miss_v = (img_val   - lab_val)   | (lab_val   - img_val)
    miss_s = (img_test  - lab_test)  | (lab_test  - img_test)

    ok = True
    if overlap_iv:
        ok = False; print("[Split] ERROR: imágenes repetidas entre splits:", sorted(list(overlap_iv))[:10], "…")
    if overlap_lv:
        ok = False; print("[Split] ERROR: labels repetidos entre splits:", sorted(list(overlap_lv))[:10], "…")
    if miss_t:
        ok = False; print("[Split] ERROR train: descalces img/label:", sorted(list(miss_t))[:10], "…")
    if miss_v:
        ok = False; print("[Split] ERROR val: descalces img/label:", sorted(list(miss_v))[:10], "…")
    if miss_s:
        ok = False; print("[Split] ERROR test: descalces img/label:", sorted(list(miss_s))[:10], "…")

    if ok:
        print("[Split] Integridad OK (sin solapamientos; imágenes y labels alineados).")
    elif strict:
        sys.exit(1)
    return ok

# =========================
# Geometría / Intersecciones
# =========================
def rect_from_xyxy(x1,y1,x2,y2): return (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))
def bbox_overlap(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    return (ax1<=bx2) and (ax2>=bx1) and (ay1<=by2) and (ay2>=by1)

def point_in_poly(x,y, poly):
    inside=False; n=len(poly)
    for i in range(n):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
        cond=((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12)+x1)
        if cond: inside=not inside
    return inside

def poly_rect_intersect(poly, rect):
    px=[p[0] for p in poly]; py=[p[1] for p in poly]
    pbb=(min(px),min(py),max(px),max(py))
    if not bbox_overlap(pbb, rect): return False
    x1,y1,x2,y2=rect
    for (x,y) in poly:
        if x1<=x<=x2 and y1<=y<=y2: return True
    for (cx,cy) in [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]:
        if point_in_poly(cx,cy, poly): return True
    return True

# =========================
# Longitud de fisura (esqueleto + fallback PCA)
# =========================
_NEI8=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def _neighbors8(y,x,H,W):
    for dy,dx in _NEI8:
        ny, nx = y+dy, x+dx
        if 0<=ny<H and 0<=nx<W:
            yield ny, nx, (1.0 if (dy==0 or dx==0) else sqrt(2.0))

def _dijkstra_prev_on_skel(sk_bin, src):
    """
    Dijkstra sobre skeleton 8-conectado.
    Retorna: (dist_dict, prev_dict, farthest_node, farthest_dist)
    prev_dict permite reconstruir camino.
    """
    H, W = sk_bin.shape
    sy, sx = int(src[0]), int(src[1])
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

    return dist, prev, far_node, float(far_dist)


def _reconstruct_path(prev, end_node):
    """Reconstruye lista [(x,y), ...] desde prev dict (nodos son (y,x))."""
    path = []
    cur = end_node
    while cur is not None:
        y, x = cur
        path.append((float(x), float(y)))
        cur = prev.get(cur, None)
    path.reverse()
    return np.array(path, dtype=float)

def skeleton_diameter_path(sk_bin):
    """
    Retorna:
      pts_xy: Nx2 (x,y) del camino central (diámetro) sobre skeleton
      L_px: longitud curvilínea en px (1 / sqrt2)
      p1_yx, p2_yx: endpoints (y,x)
    """
    H, W = sk_bin.shape
    ys, xs = np.where(sk_bin)
    if len(ys) == 0:
        return None, 0.0, None, None

    # endpoints: pixels con grado 1; si no hay, toma cualquiera
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

    # 1) desde un endpoint, ir al más lejano
    _, _, a, _ = _dijkstra_prev_on_skel(sk_bin, endpoints[0])

    # 2) desde a, ir al más lejano b (esto da el “diámetro”)
    _, prev, b, L_px = _dijkstra_prev_on_skel(sk_bin, a)

    pts_xy = _reconstruct_path(prev, b)  # camino EXACTO del diámetro
    return pts_xy, float(L_px), a, b

def _mask_from_poly(poly, W, H):
    m=Image.new('L',(W,H),0); ImageDraw.Draw(m).polygon([tuple(p) for p in poly],outline=1,fill=1)
    return np.array(m,dtype=np.uint8)

def save_mask_plot_mm(poly, pts_center_px, wsegs_px, mmx, mmy, out_png, title):
    import matplotlib.pyplot as plt
    import os

    # polígono a mm
    px = [float(p[0])*mmx for p in poly]
    py = [float(p[1])*mmy for p in poly]

    plt.figure()
    plt.fill(px, py, alpha=0.25, label="Máscara fisura")  # relleno

    # eje central a mm
    if pts_center_px is not None and len(pts_center_px) >= 2:
        xs = [float(p[0])*mmx for p in pts_center_px]
        ys = [float(p[1])*mmy for p in pts_center_px]
        plt.plot(xs, ys, '-', linewidth=2, color="blue", label="Eje central")
        plt.plot(xs, ys, 'o', markersize=2, color="blue")

    # anchos (mm)
    if wsegs_px:
        labels = ["Ancho 15%", "Ancho 50%", "Ancho 85%"]
        colors = ["orange", "magenta", "green"]
        for (seg, lab, col) in zip(wsegs_px, labels, colors):
            x1,y1,x2,y2 = seg
            plt.plot([x1*mmx, x2*mmx], [y1*mmy, y2*mmy], '-', linewidth=3, color=col, label=lab)

    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.gca().invert_yaxis()
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def _crop_to_bbox(mask, rect):
    x1,y1,x2,y2=[int(round(v)) for v in rect]
    x1=max(0,x1); y1=max(0,y1); x2=min(mask.shape[1],x2); y2=min(mask.shape[0],y2)
    if x2<=x1 or y2<=y1: return np.zeros((1,1),dtype=np.uint8),(0,0)
    return mask[y1:y2,x1:x2].copy(), (x1,y1)

def _skeletonize(mask_bin):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask_bin.astype(bool)).astype(np.uint8)
    except Exception: return None

def _longest_path_metrics_on_skeleton(sk):
    H, W = sk.shape
    ys, xs = np.where(sk > 0)
    if len(ys) == 0:
        return 0.0, None, None
    deg = {}
    for y, x in zip(ys, xs):
        d = 0
        for dy, dx in _NEI8:  # corregido: pares (dy,dx)
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and sk[ny, nx]:
                d += 1
        deg[(y, x)] = d
    endpoints = [p for p, d in deg.items() if d == 1] or [(ys[0], xs[0])]
    def dijkstra(src):
        (sy, sx) = src
        dist = {(sy, sx): 0.0}
        pq = [(0.0, sy, sx)]
        best = (0.0, (sy, sx))
        while pq:
            d, y, x = heapq.heappop(pq)
            if d > dist[(y, x)] + 1e-9: continue
            if d > best[0]: best = (d, (y, x))
            for ny, nx, w in _neighbors8(y, x, H, W):
                if sk[ny, nx] == 0: continue
                nd = d + w
                if (ny, nx) not in dist or nd < dist[(ny, nx)]:
                    dist[(ny, nx)] = nd; heapq.heappush(pq, (nd, ny, nx))
        return best
    best_len = 0.0; pbest = (None, None)
    for p in endpoints:
        far = dijkstra(p)[1]; far2 = dijkstra(far)
        if far2[0] > best_len: best_len = far2[0]; pbest = (p, far2[1])
    return float(best_len), pbest[0], pbest[1]

def _path_points_between(sk_bin, p1, p2):
    """
    sk_bin: skeleton binario (H,W) boolean
    p1,p2: puntos (y,x) en coords de imagen
    Retorna una lista Nx2 de puntos (x,y) ordenados sobre el skeleton.
    """
    H, W = sk_bin.shape
    sy, sx = int(p1[0]), int(p1[1])
    ty, tx = int(p2[0]), int(p2[1])

    # BFS simple sobre píxeles skeleton (8-conectado)
    from collections import deque
    q = deque()
    q.append((sy, sx))
    prev = { (sy, sx): None }

    while q:
        y, x = q.popleft()
        if (y, x) == (ty, tx):
            break
        for dy, dx in _NEI8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and sk_bin[ny, nx] and (ny, nx) not in prev:
                prev[(ny, nx)] = (y, x)
                q.append((ny, nx))

    if (ty, tx) not in prev:
        return None

    # reconstruir camino
    path = []
    cur = (ty, tx)
    while cur is not None:
        y, x = cur
        path.append((float(x), float(y)))
        cur = prev[cur]
    path.reverse()
    return np.array(path, dtype=float)

def _pca_metrics(mask_bin):
    ys,xs=np.where(mask_bin>0)
    if len(xs)<2: return 0.0,None,None
    pts=np.stack([xs,ys],axis=1).astype(np.float64)
    mu=pts.mean(axis=0,keepdims=True); X=pts-mu
    _,_,VT=np.linalg.svd(X,full_matrices=False)
    pc1=VT[0]; proj=X@pc1; tmin,tmax=proj.min(),proj.max()
    p1=(mu[0]+tmin*pc1); p2=(mu[0]+tmax*pc1)
    dx,dy=(p2[0]-p1[0]),(p2[1]-p1[1]); L=float(np.hypot(dx,dy))
    return L,(float(p1[1]),float(p1[0])),(float(p2[1]),float(p2[0]))

def crack_length_metrics_from_poly(poly, W, H, bbox_xyxy=None):
    mask = _mask_from_poly(poly, W, H)
    origin = (0, 0)
    if bbox_xyxy is not None:
        mask, origin = _crop_to_bbox(mask, bbox_xyxy)
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0

    # Skeleton o fallback PCA
    sk = _skeletonize(mask)
    L, p1, p2 = (0.0, None, None)
    if sk is not None and sk.any():
        L, p1, p2 = _longest_path_metrics_on_skeleton(sk > 0)
    if L < 1e-6 or p1 is None or p2 is None:
        # fallback PCA siempre que haya más de 2 píxeles
        L, p1, p2 = _pca_metrics(mask > 0)

    # Coordenadas absolutas
    if p1 is None or p2 is None:
        return 0.0, 0.0, 0.0
    oy, ox = origin[1], origin[0]
    y1, x1 = p1[0] + oy, p1[1] + ox
    y2, x2 = p2[0] + oy, p2[1] + ox

    # Longitud curvilínea (L) y distancias rectas en píxeles
    dx = (x2 - x1)
    dy = (y2 - y1)
    return float(L), float(dx), float(dy)

def crack_endpoints_from_poly(poly, W, H, bbox_xyxy=None):
    """
    Devuelve endpoints (x1,y1,x2,y2) en pixeles usando skeleton (o PCA fallback).
    """
    mask = _mask_from_poly(poly, W, H)
    origin = (0, 0)
    if bbox_xyxy is not None:
        mask, origin = _crop_to_bbox(mask, bbox_xyxy)
    if mask.sum() == 0:
        return None

    sk = _skeletonize(mask)
    p1 = p2 = None
    if sk is not None and sk.any():
        _, p1, p2 = _longest_path_metrics_on_skeleton(sk > 0)
    if p1 is None or p2 is None:
        _, p1, p2 = _pca_metrics(mask > 0)
    if p1 is None or p2 is None:
        return None

    oy, ox = origin[1], origin[0]
    y1, x1 = p1[0] + oy, p1[1] + ox
    y2, x2 = p2[0] + oy, p2[1] + ox
    return float(x1), float(y1), float(x2), float(y2)

def crack_inclination_from_endpoints(x1, y1, x2, y2):
    """
    ángulo en grados y clase: Horizontal / Vertical / Diagonal
    """
    dx = (x2 - x1)
    dy = (y2 - y1)
    ang = math.degrees(math.atan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0
    a = abs(ang) % 180.0
    if a > 90.0:
        a = 180.0 - a

    # umbrales simples (ajustables)
    if a <= 15.0:
        cls = "Horizontal"
    elif a >= 75.0:
        cls = "Vertical"
    else:
        cls = "Diagonal"
    return float(ang), cls

def crack_zone_in_beam(x, y, bx1, by1, bx2, by2):
    """
    Localización relativa dentro del bbox de viga.
    Retorna (zona_vertical, zona_horizontal).
    """
    w = max(1e-9, (bx2 - bx1))
    h = max(1e-9, (by2 - by1))
    nx = (x - bx1) / w
    ny = (y - by1) / h

    # vertical: superior/central/inferior
    if ny < 1/3:
        zv = "Superior"
    elif ny < 2/3:
        zv = "Central"
    else:
        zv = "Inferior"

    # horizontal: izquierda/centro/derecha
    if nx < 1/3:
        zh = "Izquierda"
    elif nx < 2/3:
        zh = "Centro"
    else:
        zh = "Derecha"

    return zv, zh


def orient_centerline_bottom_to_top(pts_xy):
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


def crack_widths_three_stations_from_poly_curvilinear(poly, W, H, stations=(0.15, 0.50, 0.85), max_step=600):
    """
    Estaciones por longitud CURVILÍNEA sobre el eje central (skeleton diameter path).
    Devuelve: widths_px, segs_px, centers_px
    """
    mask = _mask_from_poly(poly, W, H).astype(bool)
    if mask.sum() == 0:
        return [0.0, 0.0, 0.0], [(0,0,0,0)]*3, [(0,0)]*3

    # 1) obtener eje central (pts) como polilínea (x,y)
    sk = _skeletonize(mask.astype(np.uint8))
    pts = None
    if sk is not None and sk.any():
        out = skeleton_diameter_path(sk > 0)
        if out is not None:
            pts, _, _, _ = out

    # fallback: PCA (recta)
    if pts is None or len(pts) < 2:
        _, p1, p2 = _pca_metrics(mask)
        if p1 is None or p2 is None:
            return [0.0, 0.0, 0.0], [(0,0,0,0)]*3, [(0,0)]*3
        pts = np.array([[p1[1], p1[0]], [p2[1], p2[0]]], dtype=float)  # (x,y)

    # Convención física fija:
    #   0.15 -> inferior
    #   0.50 -> intermedio
    #   0.85 -> superior
    pts = orient_centerline_bottom_to_top(pts)

    # 2) arclength acumulado
    dxy = np.diff(pts, axis=0)
    seglen = np.hypot(dxy[:,0], dxy[:,1])
    total = float(seglen.sum())
    if total < 1e-9:
        cx, cy = float(pts[0,0]), float(pts[0,1])
        return [0.0,0.0,0.0], [(cx,cy,cx,cy)]*3, [(cx,cy)]*3

    cum = np.concatenate([[0.0], np.cumsum(seglen)])

    def point_and_tangent_at(tfrac):
        target = tfrac * total
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = max(0, min(j, len(pts)-2))
        t0 = cum[j]; t1 = cum[j+1]
        alpha = 0.0 if (t1-t0) < 1e-9 else (target - t0) / (t1 - t0)
        p = (1-alpha)*pts[j] + alpha*pts[j+1]
        tang = pts[j+1] - pts[j]
        nrm = float(np.hypot(tang[0], tang[1]))
        if nrm < 1e-9:
            tang = np.array([1.0, 0.0])
        else:
            tang = tang / nrm
        # normal (perp)
        normal = np.array([-tang[1], tang[0]])
        return (float(p[0]), float(p[1])), (float(normal[0]), float(normal[1]))

    Hh, Ww = mask.shape

    def inb(ix, iy):
        return 0 <= ix < Ww and 0 <= iy < Hh

    widths = []
    segs = []
    centers = []

    for tf in stations:
        (cx, cy), (nx, ny) = point_and_tangent_at(tf)
        centers.append((cx, cy))

        # avanzar +
        ppx, ppy = cx, cy
        for _ in range(max_step):
            ix, iy = int(round(ppx)), int(round(ppy))
            if (not inb(ix, iy)) or (not mask[iy, ix]):
                break
            ppx += nx
            ppy += ny
        x_pos, y_pos = ppx, ppy

        # avanzar -
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

    return widths, segs, centers

def _line_width_segment(center_xy, dx_px, dy_px, half_len_px):
    """Segmento (x1,y1,x2,y2) perpendicular al vector (dx,dy), centrado en center_xy."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    vx, vy = float(dx_px), float(dy_px)
    # normal
    nx, ny = -vy, vx
    nrm = math.hypot(nx, ny)
    if nrm < 1e-9:
        nx, ny = 1.0, 0.0
    else:
        nx, ny = nx / nrm, ny / nrm
    x1 = cx - nx * half_len_px
    y1 = cy - ny * half_len_px
    x2 = cx + nx * half_len_px
    y2 = cy + ny * half_len_px
    return float(x1), float(y1), float(x2), float(y2)

def safe_import_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
        return True
    except Exception as e:
        print("\n[ERROR] No se pudo importar 'ultralytics'. Instala con:  pip install ultralytics")
        print("Detalle:", e); return False


def folder_has_valid_images(folder):
    if not folder or not os.path.isdir(folder):
        return False
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jp2", ".avif", ".heic", ".heif", ".dng", ".mpo", ".jpeg2000"}
    try:
        for name in os.listdir(folder):
            fp = os.path.join(folder, name)
            if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in valid_ext:
                return True
    except Exception:
        return False
    return False

# =========================
# Train + Val + Pred + Infer
# =========================
def train_val_predict_all(yaml_path, run_project, run_name, device, epochs, batch, imgsz, workers,
                          model_name, patience, optimizer, lr0, lrf,
                          mosaic, mixup, copy_paste, hsv_s, scale, fliplr, plots,
                          degrees, translate, shear, perspective, auto_augment, erasing, flipud,
                          predict_source):
    from ultralytics import YOLO

    print("\n=== ENTRENANDO YOLOv11 (segment Fisura/Apoyo) ===")
    model = YOLO(model_name)
    r = model.train(
        data=yaml_path, imgsz=imgsz, epochs=epochs, batch=batch, device=device, workers=workers,
        patience=patience, project=run_project, name=run_name,
        optimizer=optimizer, lr0=lr0, lrf=lrf,
        mosaic=mosaic, mixup=mixup, copy_paste=copy_paste,
        hsv_s=hsv_s, scale=scale, fliplr=fliplr, flipud=flipud,
        degrees=degrees, translate=translate, shear=shear, perspective=perspective,
        auto_augment=auto_augment, erasing=erasing, plots=plots
    )
    save_dir = str(getattr(r, "save_dir", "")) or None
    if not save_dir:
        candidates = glob(os.path.join(run_project, "*", "weights"))
        if not candidates: raise FileNotFoundError("No se encontró ninguna carpeta con pesos en runs/**/weights")
        save_dir = os.path.dirname(max(candidates, key=os.path.getmtime))
    weights_dir = os.path.join(save_dir, "weights")
    best_pt = os.path.join(weights_dir, "best.pt")
    last_pt = os.path.join(weights_dir, "last.pt")
    if not os.path.exists(best_pt):
        if os.path.exists(last_pt):
            print("[Aviso] best.pt no existe; usando last.pt"); best_pt = last_pt
        else:
            cands = glob(os.path.join(run_project, "*", "weights", "best.pt")) + \
                    glob(os.path.join(run_project, "*", "weights", "last.pt"))
            if not cands: raise FileNotFoundError("No se encontró best.pt/last.pt tras el entrenamiento.")
            best_pt = max(cands, key=os.path.getmtime); save_dir = os.path.dirname(os.path.dirname(best_pt))

    run_dir = save_dir
    print(f"[Run] Carpeta del entrenamiento: {run_dir}")
    print(f"[Run] Pesos: {best_pt}")

    # Validación con plots
    best_model = YOLO(best_pt)
    best_model.val(data=yaml_path, imgsz=imgsz, device=device, plots=True,
                   project=run_project, name=f"{os.path.basename(run_dir)}_val_final")

# Predicciones val/test visuales omitidas para evitar consumo excesivo de RAM en CPU.
    base = os.path.dirname(os.path.dirname(yaml_path))
    for split in ["val", "test"]:
        src = os.path.join(base, "images", split)
        if folder_has_valid_images(src):
            print(f"[Predict] Omitiendo predicción visual de {split} para evitar saturación de RAM en CPU.")
        else:
            print(f"[Predict] Omitiendo {split}: no hay imágenes válidas en {src}")

    # Inferencia NUEVA (visuales) + preview índice
    if predict_source and folder_has_valid_images(predict_source):
        infer_name = f"{os.path.basename(run_dir)}_new"
        preds = best_model.predict(source=predict_source,imgsz=imgsz,device=device,project=run_project,name=infer_name,save=False,verbose=False)
        
        print("[Inferencia] Predicciones calculadas sin guardar imágenes visuales para evitar saturación de RAM.")

        print("Inferencia guardada en:", os.path.join(run_project, infer_name))
        if preds:
            try:
                raw = input("¿Qué índice de imagen quieres mostrar? (ej. 49 para la #50): ")
                idx = int(raw)
            except Exception:
                idx = 0
            idx = max(0, min(len(preds)-1, idx))
            try:
                preds[idx].show()
                print(f"Mostrando predicción índice {idx}.")
            except Exception as e:
                print("No se pudo abrir .show(). Detalle:", e)
        else:
            print("[Inferencia] No se generaron predicciones visuales para la carpeta de entrada.")
    elif predict_source:
        print(f"[Inferencia] Omitida: no hay imágenes válidas en {predict_source}")

    return best_pt, run_dir

# =========================
# Cargar bboxes de viga
# =========================
def load_beam_boxes_from_label(txt_path, W, H):
    boxes=[]
    try:
        with open(txt_path,"r",encoding="utf-8") as f:
            for line in f:
                vals=line.strip().split()
                if not vals: continue
                if len(vals)>=5 and len(vals)%2==1:  # yolo-seg
                    coords=list(map(float, vals[1:]))
                    xs=coords[0::2]; ys=coords[1::2]
                    xs=[x*W for x in xs]; ys=[y*H for y in ys]
                    boxes.append([min(xs),min(ys),max(xs),max(ys)])
                elif len(vals)==5:  # yolo-bbox
                    _,cx,cy,w,h=map(float, vals)
                    x1=(cx-w/2.0)*W; y1=(cy-h/2.0)*H
                    x2=(cx+w/2.0)*W; y2=(cy+h/2.0)*H
                    boxes.append([x1,y1,x2,y2])
    except Exception:
        pass
    return boxes

# =========================
# Segmentación (forzada 0=Fisura, 1=Apoyo)
# =========================
def segment_objects(seg_model, img_path, conf=0.25, device="cpu"):
    """Devuelve polígonos por clase.

    out = {"fisura": [...], "apoyo": [...], "viga": [...]}

    - Si el modelo solo tiene 2 clases (fisura/apoyo), "viga" queda vacío.
    - Detecta por nombre de clase (contiene 'fisur', 'apoyo/support', 'viga/beam').
    """
    r = seg_model.predict(source=img_path, conf=conf, verbose=False, save=False, device=device)[0]
    out = {"fisura": [], "apoyo": [], "viga": []}
    if getattr(r, "masks", None) is None:
        return out

    cls = r.boxes.cls.cpu().numpy().astype(int)
    polys = r.masks.xy

    # Mapa id->nombre (robusto)
    try:
        names = {int(i): str(n).lower() for i, n in r.names.items()}
    except Exception:
        try:
            names = {int(i): str(n).lower() for i, n in seg_model.model.names.items()}
        except Exception:
            names = {int(i): str(i) for i in set(cls)}

    for c, poly in zip(cls, polys):
        lab = names.get(int(c), "")
        if "fisur" in lab:
            out["fisura"].append(poly)
        elif "apoyo" in lab or "support" in lab:
            out["apoyo"].append(poly)
        elif "viga" in lab or "beam" in lab:
            out["viga"].append(poly)
        else:
            # clase desconocida -> ignora
            pass
    return out
def draw_overlay(ip, beam_bboxes, fis_polys, apo_polys, flags, outp):
    img=Image.open(ip).convert("RGB"); dr=ImageDraw.Draw(img,"RGBA")
    for poly in fis_polys: dr.polygon([tuple(p) for p in poly], fill=(255,0,0,60), outline=(255,0,0,180))
    for poly in apo_polys: dr.polygon([tuple(p) for p in poly], fill=(0,255,0,60), outline=(0,255,0,180))
    for i,b in enumerate(beam_bboxes):
        x1,y1,x2,y2=map(float,b)
        color=(255,128,0,200) if (flags and flags[i]==1) else (0,128,255,200)
        dr.rectangle([x1,y1,x2,y2], outline=color, width=3)
        dr.text((x1+5,y1+5), "Viga Fisurada" if (flags and flags[i]==1) else "Viga No Fisurada", fill=(255,255,255,255))
    os.makedirs(os.path.dirname(outp), exist_ok=True); img.save(outp)

def _save_crack_txt(out_dir, stem, rows):
    os.makedirs(out_dir, exist_ok=True)
    p=os.path.join(out_dir, f"{stem}.txt")
    with open(p,"w",encoding="utf-8") as f:
        for (i,L,dx,dy) in rows: f.write(f"{i} {L:.3f} {dx:.3f} {dy:.3f}\n")
    return p

def _append_rows_or_zero(rows_list, image_relpath, seg_rows):
    """
    rows_list: lista global (val/test/inf)
    image_relpath: ruta relativa de la imagen
    seg_rows: lista de diccionarios ya armados por segmento

    - Si NO hay fisuras -> una fila dummy con crack_idx=0, seg_idx=0.
    - Si sí hay fisuras -> se agregan todas las filas de segmentos.
    """
    if not seg_rows:
        rows_list.append({
            "image": image_relpath,
            "crack_idx": 0,
            "seg_idx": 0,
            "length_px": 0.0,
            "dx_px": 0.0,
            "dy_px": 0.0,
        })
    else:
        rows_list.extend(seg_rows)

def segment_rows_from_poly(image_relpath, crack_idx, poly, W=None, H=None):
    """
    poly viene de r.masks.xy => es CONTORNO (polígono).
    Aquí lo convertimos a LÍNEA CENTRAL (skeleton) y exportamos segmentos sobre esa línea.
    Además guardamos coordenadas absolutas para graficar bien.
    """
    rows = []

    if poly is None or len(poly) < 3 or W is None or H is None:
        rows.append({
            "image": image_relpath,
            "crack_idx": int(crack_idx),
            "seg_idx": 0,
            "length_px": 0.0,
            "dx_px": 0.0,
            "dy_px": 0.0,
            "x1_px": 0.0, "y1_px": 0.0,
            "x2_px": 0.0, "y2_px": 0.0,
        })
        return rows

    # 1) máscara binaria del polígono
    mask = _mask_from_poly(poly, W, H)
    if mask.sum() == 0:
        rows.append({
            "image": image_relpath,
            "crack_idx": int(crack_idx),
            "seg_idx": 0,
            "length_px": 0.0,
            "dx_px": 0.0,
            "dy_px": 0.0,
            "x1_px": 0.0, "y1_px": 0.0,
            "x2_px": 0.0, "y2_px": 0.0,
        })
        return rows

    # 2) skeleton
    # 2) skeleton + camino central REAL (diámetro)
    sk = _skeletonize(mask)
    pts = None

    if sk is not None and sk.any():
        out = skeleton_diameter_path(sk > 0)
        if out is not None:
            pts, Lsk, a, b = out  # pts: Nx2 (x,y)
    # fallback si skeleton falla
    if pts is None or len(pts) < 2:
        _, p1, p2 = _pca_metrics(mask > 0)
        if p1 is None or p2 is None:
            pts = None
        else:
            pts = np.array([[p1[1], p1[0]], [p2[1], p2[0]]], dtype=float)

    if pts is None or len(pts) < 2:
        return rows
    if pts is None or len(pts) < 2:
        return rows

    # 5) exportar segmentos
    for s in range(len(pts) - 1):
        x1, y1 = pts[s]
        x2, y2 = pts[s + 1]
        dx = x2 - x1
        dy = y2 - y1
        L  = float(math.hypot(dx, dy))

        rows.append({
            "image": image_relpath,
            "crack_idx": int(crack_idx),
            "seg_idx": int(s + 1),
            "length_px": L,
            "dx_px": float(dx),
            "dy_px": float(dy),
            "x1_px": float(x1), "y1_px": float(y1),
            "x2_px": float(x2), "y2_px": float(y2),
        })

    return rows

def _write_lengths_csv(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        # Escribe encabezado mínimo
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image","crack_idx","length_px","dx_px","dy_px"])
            w.writeheader()
        return csv_path

    # Usa todas las llaves presentes en la primera fila para el header
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return csv_path


def load_scale_resources(scale_csv_path):
    """
    Carga la escala global por imagen y, si existe, un perfil local de altura de viga
    exportado por dect_fis_col.py en beam_scale_profile.csv.
    """
    scale_map = {}
    profile_map = {}
    if not scale_csv_path or not os.path.exists(scale_csv_path):
        print(f"[Scale] Aviso: no existe CSV de escala: {scale_csv_path}")
        return scale_map, profile_map

    try:
        with open(scale_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get("image") or row.get("file") or row.get("filename")
                if not img:
                    continue
                key = os.path.basename(img.strip())
                try:
                    mmx = float(row.get("mm_per_px_x", "1.0"))
                    mmy = float(row.get("mm_per_px_y", "1.0"))
                except Exception:
                    mmx, mmy = 1.0, 1.0
                scale_map[key] = (mmx, mmy)
        print(f"[Scale] Escalas globales leídas para {len(scale_map)} imágenes desde {scale_csv_path}")
    except Exception as e:
        print("[Scale] Error leyendo CSV de escala:", e)

    profile_csv = os.path.join(os.path.dirname(scale_csv_path), "beam_scale_profile.csv")
    if os.path.exists(profile_csv):
        try:
            with open(profile_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img = row.get("image") or row.get("file") or row.get("filename")
                    if not img:
                        continue
                    key = os.path.basename(img.strip())
                    try:
                        profile_map.setdefault(key, []).append({
                            "u_pct": float(row.get("u_pct", "0")),
                            "local_height_px": float(row.get("local_height_px", "0")),
                            "mm_per_px_y_local": float(row.get("mm_per_px_y_local", row.get("mm_per_px_y", "1.0"))),
                        })
                    except Exception:
                        continue
            for key in profile_map:
                profile_map[key].sort(key=lambda d: d["u_pct"])
            print(f"[Scale] Perfil local leído para {len(profile_map)} imágenes desde {profile_csv}")
        except Exception as e:
            print("[Scale] Error leyendo perfil local de escala:", e)
    else:
        print(f"[Scale] No se encontró perfil local: {profile_csv}. Se usará mm_per_px_y global.")

    return scale_map, profile_map


def lookup_local_scale_y(profile_map, image_key, u_pct, default_mmy, window_pct=5.0):
    rows = profile_map.get(os.path.basename(image_key).strip(), [])
    if not rows:
        return float(default_mmy), 0.0, float(u_pct), float(window_pct)
    lo = float(u_pct) - float(window_pct)
    hi = float(u_pct) + float(window_pct)
    vals = [r for r in rows if lo <= float(r.get("u_pct", 0.0)) <= hi and float(r.get("mm_per_px_y_local", 0.0)) > 0]
    if not vals:
        vals = rows
    chosen = max(vals, key=lambda r: float(r.get("mm_per_px_y_local", 0.0))) if vals else None
    if not chosen:
        return float(default_mmy), 0.0, float(u_pct), float(window_pct)
    return float(chosen.get("mm_per_px_y_local", default_mmy)), float(chosen.get("local_height_px", 0.0)), float(u_pct), float(window_pct)


def width_segment_mm(seg_px, mmx, mmy):
    x1, y1, x2, y2 = seg_px
    return float(math.hypot((x2 - x1) * mmx, (y2 - y1) * mmy))


def segment_length_mm_with_row_scale(rows):
    total = 0.0
    for r in rows:
        dx = float(r.get("dx_px", 0.0) or 0.0)
        dy = float(r.get("dy_px", 0.0) or 0.0)
        mmx = float(r.get("mm_per_px_x", 1.0) or 1.0)
        mmy = float(r.get("mm_per_px_y", 1.0) or 1.0)
        total += math.hypot(dx * mmx, dy * mmy)
    return float(total)


def apply_mm_scale_to_rows(rows, scale_map):
    """
    Usa primero la escala que ya tenga cada fila (p. ej. mm_per_px_y local por fisura).
    Si no existe, cae al scale_map global por imagen.
    """
    for r in rows:
        img_key = os.path.basename(r.get("image", "")).strip()
        mmx0, mmy0 = scale_map.get(img_key, (1.0, 1.0))
        mmx = float(r.get("mm_per_px_x", mmx0) or mmx0)
        mmy = float(r.get("mm_per_px_y", mmy0) or mmy0)

        L_px = float(r.get("length_px", 0.0) or 0.0)
        dx   = float(r.get("dx_px", 0.0) or 0.0)
        dy   = float(r.get("dy_px", 0.0) or 0.0)

        diag_px = math.hypot(dx, dy)
        dx_mm = dx * mmx
        dy_mm = dy * mmy
        diag_mm = math.hypot(dx_mm, dy_mm)

        if diag_px > 1e-6 and diag_mm > 0:
            factor = diag_mm / diag_px
            L_mm = L_px * factor
        else:
            L_mm = L_px * ((mmx + mmy) / 2.0)

        r["mm_per_px_x"] = mmx
        r["mm_per_px_y"] = mmy
        r["dx_mm"] = dx_mm
        r["dy_mm"] = dy_mm
        r["length_mm"] = L_mm

def compute_local_angles_mm(rows):
    """
    Agrega columnas de ángulo local en milímetros por segmento.
    rows: lista de diccionarios (cada fila corresponde a una fisura)
    """
    import math

    for r in rows:
        try:
            dx_mm = float(r.get('dx_mm', 0))
            dy_mm = float(r.get('dy_mm', 0))
        except:
            continue

        angle = math.degrees(math.atan2(dy_mm, dx_mm))  # atan2 maneja signo y cuadrante
        r['angle_deg_mm'] = angle

def plot_crack_geometry_mm(image_name, xs_mm, ys_mm, out_path, width_segs_mm=None):
    import matplotlib.pyplot as plt
    import os

    plt.figure()

    # Longitud (eje central) SIEMPRE azul
    plt.plot(xs_mm, ys_mm, '-', linewidth=2, color="blue", label="Eje central")
    plt.plot(xs_mm, ys_mm, 'o', markersize=2, color="blue")

    plt.title(image_name)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.gca().invert_yaxis()

    # Anchos: colores distintos por estación (15%, 50%, 85%)
    if width_segs_mm:
        labels = ["Ancho 15%", "Ancho 50%", "Ancho 85%"]
        colors = ["orange", "magenta", "green"]
        for (seg, lab, col) in zip(width_segs_mm, labels, colors):
            x1,y1,x2,y2 = seg
            plt.plot([x1,x2], [y1,y2], '-', linewidth=3, color=col, label=lab)

        plt.legend(loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

from pathlib import Path  # si no lo tienes ya arriba

def process_crack_segments_mm(rows, run_dir: Path, plots_subdir: str) -> None:
    from collections import defaultdict

    plots_dir = (Path(run_dir) / "beam_class" / plots_subdir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for r in rows:
        img = r.get("image", "")
        cid = int(r.get("crack_idx", 0) or 0)
        groups[(img, cid)].append(r)

    for (img, cid), rr in groups.items():
        rr = sorted(rr, key=lambda x: int(x.get("seg_idx", 0) or 0))

        # reconstrucción de la polilínea a partir de dx_mm/dy_mm acumulados
        mmx0 = float(rr[0].get("mm_per_px_x", 1.0) or 1.0)
        mmy0 = float(rr[0].get("mm_per_px_y", 1.0) or 1.0)

        xs_mm = [float(rr[0].get("x1_px", 0.0) or 0.0) * mmx0]
        ys_mm = [float(rr[0].get("y1_px", 0.0) or 0.0) * mmy0]

        for r in rr:
            mmx = float(r.get("mm_per_px_x", 1.0) or 1.0)
            mmy = float(r.get("mm_per_px_y", 1.0) or 1.0)

            xs_mm.append(float(r.get("x2_px", 0.0) or 0.0) * mmx)
            ys_mm.append(float(r.get("y2_px", 0.0) or 0.0) * mmy)
        if len(xs_mm) < 2:
            continue

        # extraer los 3 segmentos de ancho (una sola vez por fisura)
        r0 = rr[0]
        width_segs_mm = None
        if "w_top_x1_mm" in r0:
            width_segs_mm = [
                (float(r0.get("w_top_x1_mm",0)), float(r0.get("w_top_y1_mm",0)), float(r0.get("w_top_x2_mm",0)), float(r0.get("w_top_y2_mm",0))),
                (float(r0.get("w_mid_x1_mm",0)), float(r0.get("w_mid_y1_mm",0)), float(r0.get("w_mid_x2_mm",0)), float(r0.get("w_mid_y2_mm",0))),
                (float(r0.get("w_bot_x1_mm",0)), float(r0.get("w_bot_y1_mm",0)), float(r0.get("w_bot_x2_mm",0)), float(r0.get("w_bot_y2_mm",0))),
            ]

        out_png = plots_dir / f"{Path(img).stem}_crack{cid}.png"
        plot_crack_geometry_mm(f"{img} – crack {cid}", xs_mm, ys_mm, str(out_png), width_segs_mm=width_segs_mm)

# =========================
# Clasificación + Longitudes (val/test) y CSVs
# =========================
from pathlib import Path  # déjalo solo una vez en el archivo, arriba o aquí

def classify_val_test_with_labels(run_dir, ds_root, seg_model, conf_seg=0.25, scale_map=None, profile_map=None, device="cpu"):
    out_root = os.path.join(run_dir, "beam_class")
    os.makedirs(out_root, exist_ok=True)
    vis_dir = os.path.join(out_root, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    txt_dir = os.path.join(out_root, "crack_lengths_txt")
    os.makedirs(txt_dir, exist_ok=True)

    csv_class   = os.path.join(out_root, "beam_classification.csv")
    csv_len_val = os.path.join(out_root, "crack_lengths_val.csv")
    csv_len_test= os.path.join(out_root, "crack_lengths_test.csv")

    class_rows   = []
    len_rows_val = []
    len_rows_test = []

    for split in ("val", "test"):
        img_dir = os.path.join(ds_root, "images", split)
        det_dir = os.path.join(ds_root, "labels_det", split)

        if not os.path.isdir(img_dir):
            print(f"[CLASS] Saltando {split}: no existe images/{split}.")
            continue

        imgs = []
        for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
            imgs += glob(os.path.join(img_dir, e))
        imgs = sorted(imgs)
        if not imgs:
            print(f"[CLASS] {split}: sin imágenes.")
            continue

        for ip in imgs:
            stem = os.path.splitext(os.path.basename(ip))[0]
            try:
                im = Image.open(ip).convert("RGB")
                W, H = im.size
            except Exception:
                rel = os.path.relpath(ip, ds_root)
                if split == "val":
                    _append_rows_or_zero(len_rows_val, rel, [])
                else:
                    _append_rows_or_zero(len_rows_test, rel, [])
                continue

            seg = segment_objects(seg_model, ip, conf=conf_seg, device=device)
            fis_polys, apo_polys = seg["fisura"], seg["apoyo"]

            # beams (para zonas dentro de la viga)
            rel = os.path.relpath(ip, ds_root)
            ldet = os.path.join(det_dir, stem + ".txt")
            beams = load_beam_boxes_from_label(ldet, W, H) if os.path.exists(ldet) else []

            # escala global por imagen + perfil local por fisura
            img_key = os.path.basename(rel).strip()
            mmx, mmy_global = (scale_map or {}).get(img_key, (1.0, 1.0))

            # --- Longitudes por SEGMENTO + métricas por fisura ---
            crack_seg_rows = []
            crack_idx = 1

            for poly in fis_polys:
                seg_rows = segment_rows_from_poly(rel, crack_idx, poly, W=W, H=H)

                # (A) extremos + inclinación
                ep = crack_endpoints_from_poly(poly, W, H, bbox_xyxy=None)
                if ep is None:
                    x1=y1=x2=y2=0.0
                    ang_deg = 0.0
                    ang_cls = "Horizontal"
                else:
                    x1, y1, x2, y2 = ep
                    ang_deg, ang_cls = crack_inclination_from_endpoints(x1, y1, x2, y2)

                # (B) anchos (3 estaciones)
                widths_px, wsegs_px, centers_px = crack_widths_three_stations_from_poly_curvilinear(poly, W, H)

                # (C) asignar a una viga + zona (si hay labels_det) y localizar la fisura al 50% de su eje
                beam_idx = -1
                zv = "NA"
                zh = "NA"
                crack_u_pct = 50.0
                mxp = 0.5*(x1+x2)
                myp = 0.5*(y1+y2)
                if beams:
                    for bi, b in enumerate(beams):
                        bx1, by1, bx2, by2 = rect_from_xyxy(*b)
                        if poly_rect_intersect(poly, (bx1, by1, bx2, by2)):
                            beam_idx = bi
                            zv, zh = crack_zone_in_beam(mxp, myp, bx1, by1, bx2, by2)
                            crack_u_pct = 100.0 * max(0.0, min(1.0, (mxp - bx1) / max(1e-9, (bx2 - bx1))))
                            break

                local_mmy, local_height_px, crack_u_pct, window_pct = lookup_local_scale_y(
                    profile_map or {}, img_key, crack_u_pct, mmy_global, window_pct=5.0
                )
                widths_mm = [
                    width_segment_mm(wsegs_px[0], mmx, local_mmy),
                    width_segment_mm(wsegs_px[1], mmx, local_mmy),
                    width_segment_mm(wsegs_px[2], mmx, local_mmy),
                ]

                # (D) inyectar en todas las filas del CSV de segmentos
                for r in seg_rows:
                    r["mm_per_px_x"] = float(mmx)
                    r["mm_per_px_y"] = float(local_mmy)
                    r["mm_per_px_y_global"] = float(mmy_global)
                    r["local_height_px"] = float(local_height_px)
                    r["local_u_pct"] = float(crack_u_pct)
                    r["local_window_pct"] = float(window_pct)

                    r["end_x1_px"] = float(x1); r["end_y1_px"] = float(y1)
                    r["end_x2_px"] = float(x2); r["end_y2_px"] = float(y2)
                    r["inclination_deg"] = float(ang_deg)
                    r["inclination_class"] = ang_cls

                    r["w_top_px"] = float(widths_px[0]); r["w_mid_px"] = float(widths_px[1]); r["w_bot_px"] = float(widths_px[2])
                    r["w_top_mm"] = float(widths_mm[0]); r["w_mid_mm"] = float(widths_mm[1]); r["w_bot_mm"] = float(widths_mm[2])

                    (ax1,ay1,ax2,ay2) = wsegs_px[0]
                    (bx1_,by1_,bx2_,by2_) = wsegs_px[1]
                    (cx1,cy1,cx2,cy2) = wsegs_px[2]

                    r["w_top_x1_mm"] = float(ax1)*mmx; r["w_top_y1_mm"] = float(ay1)*local_mmy
                    r["w_top_x2_mm"] = float(ax2)*mmx; r["w_top_y2_mm"] = float(ay2)*local_mmy

                    r["w_mid_x1_mm"] = float(bx1_)*mmx; r["w_mid_y1_mm"] = float(by1_)*local_mmy
                    r["w_mid_x2_mm"] = float(bx2_)*mmx; r["w_mid_y2_mm"] = float(by2_)*local_mmy

                    r["w_bot_x1_mm"] = float(cx1)*mmx; r["w_bot_y1_mm"] = float(cy1)*local_mmy
                    r["w_bot_x2_mm"] = float(cx2)*mmx; r["w_bot_y2_mm"] = float(cy2)*local_mmy

                    r["beam_idx"] = int(beam_idx)
                    r["zone_v"] = zv
                    r["zone_h"] = zh

                crack_seg_rows.extend(seg_rows)
                crack_idx += 1

            # guardar filas de esta imagen en val/test
            if split == "val":
                _append_rows_or_zero(len_rows_val, rel, crack_seg_rows)
            else:
                _append_rows_or_zero(len_rows_test, rel, crack_seg_rows)

            # Clasificación por viga
            ldet = os.path.join(det_dir, stem + ".txt")
            beams = load_beam_boxes_from_label(ldet, W, H) if os.path.exists(ldet) else []
            flags = []
            if beams:
                sums  = [0.0 for _ in beams]
                hits  = []
                for i, b in enumerate(beams):
                    bx1, by1, bx2, by2 = rect_from_xyxy(*b)
                    k = 0
                    for poly in fis_polys:
                        if poly_rect_intersect(poly, (bx1, by1, bx2, by2)):
                            L, dx, dy = crack_length_metrics_from_poly(poly, W, H, (bx1, by1, bx2, by2))
                            if L < 1e-9:
                                L = 0.0; dx = 0.0; dy = 0.0
                            sums[i] += L
                            k += 1
                    hits.append(k)
                    flags.append(1 if k > 0 else 0)

                for i, b in enumerate(beams):
                    class_rows.append({
                        "image": rel,
                        "beam_idx": i,
                        "bbox_x1": float(b[0]), "bbox_y1": float(b[1]),
                        "bbox_x2": float(b[2]), "bbox_y2": float(b[3]),
                        "is_cracked": int(flags[i]),
                        "beam_class": "Viga Fisurada" if flags[i] == 1 else "Viga No Fisurada",
                        "num_crack_masks_touching": int(hits[i]),
                        "num_support_masks_total": int(len(apo_polys)),
                        "crack_length_px_sum": float(sums[i]),
                    })

            try:
                draw_overlay(
                    ip, beams, fis_polys, apo_polys,
                    flags if beams else None,
                    os.path.join(vis_dir, f"{stem}.jpg")
                )
            except Exception:
                pass

    # === Escala + ángulos SIEMPRE (aunque scale_map venga vacío) ===
    # apply_mm_scale_to_rows hace fallback a (1.0, 1.0) si la imagen no está en el mapa.
    apply_mm_scale_to_rows(len_rows_val,  scale_map)
    apply_mm_scale_to_rows(len_rows_test, scale_map)

    compute_local_angles_mm(len_rows_val)
    compute_local_angles_mm(len_rows_test)

    # Los plots solo necesitan alguna escala; si no hay, se asume 1 mm/px.
    process_crack_segments_mm(len_rows_val,  Path(run_dir), "plots_mm_val")
    process_crack_segments_mm(len_rows_test, Path(run_dir), "plots_mm_test")
    
    # Escritura de CSVs
    with open(csv_class, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image","beam_idx","bbox_x1","bbox_y1","bbox_x2","bbox_y2",
            "is_cracked","beam_class",
            "num_crack_masks_touching","num_support_masks_total",
            "crack_length_px_sum"
        ])
        w.writeheader()
        w.writerows(class_rows)

    _write_lengths_csv(csv_len_val,  len_rows_val)
    _write_lengths_csv(csv_len_test, len_rows_test)

    print("[CLASS] CSV clasificación:", csv_class)
    print("[CLASS] CSV longitudes val:",  csv_len_val)
    print("[CLASS] CSV longitudes test:", csv_len_test)
    print("[CLASS] VIS:", vis_dir, " | TXT:", txt_dir)

# =========================
# Longitudes por INFERENCIA (CSV)
# =========================
def export_infer_lengths_csv(seg_model, src_folder, out_csv_path, conf_seg=0.25, scale_map=None, profile_map=None, run_dir=None, device="cpu"):
    imgs = []
    for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
        imgs += glob(os.path.join(src_folder, e))
    imgs = sorted(imgs)

    rows = []
    for ip in imgs:
        try:
            im = Image.open(ip).convert("RGB")
            W, H = im.size
        except Exception:
            _append_rows_or_zero(rows, os.path.basename(ip), [])
            continue

        seg = segment_objects(seg_model, ip, conf=conf_seg, device=device)
        fis_polys = seg["fisura"]

        rel = os.path.basename(ip)

        # escala por imagen
        mmx, mmy_global = (scale_map or {}).get(rel.strip(), (1.0, 1.0))

        crack_seg_rows = []
        crack_idx = 1
        for poly in fis_polys:
            seg_rows = segment_rows_from_poly(rel, crack_idx, poly, W=W, H=H)

            ep = crack_endpoints_from_poly(poly, W, H, bbox_xyxy=None)
            if ep is None:
                x1=y1=x2=y2=0.0
                ang_deg = 0.0
                ang_cls = "Horizontal"
            else:
                x1, y1, x2, y2 = ep
                ang_deg, ang_cls = crack_inclination_from_endpoints(x1, y1, x2, y2)

            widths_px, wsegs_px, _ = crack_widths_three_stations_from_poly_curvilinear(poly, W, H)
            crack_u_pct = 100.0 * max(0.0, min(1.0, (0.5*(x1+x2)) / max(1.0, float(W))))
            local_mmy, local_height_px, crack_u_pct, window_pct = lookup_local_scale_y(
                profile_map or {}, rel.strip(), crack_u_pct, mmy_global, window_pct=5.0
            )
            widths_mm = [
                width_segment_mm(wsegs_px[0], mmx, local_mmy),
                width_segment_mm(wsegs_px[1], mmx, local_mmy),
                width_segment_mm(wsegs_px[2], mmx, local_mmy),
            ]

            for r in seg_rows:
                r["mm_per_px_x"] = float(mmx)
                r["mm_per_px_y"] = float(local_mmy)
                r["mm_per_px_y_global"] = float(mmy_global)
                r["local_height_px"] = float(local_height_px)
                r["local_u_pct"] = float(crack_u_pct)
                r["local_window_pct"] = float(window_pct)

                r["end_x1_px"] = float(x1); r["end_y1_px"] = float(y1)
                r["end_x2_px"] = float(x2); r["end_y2_px"] = float(y2)
                r["inclination_deg"] = float(ang_deg)
                r["inclination_class"] = ang_cls

                r["w_top_px"] = float(widths_px[0]); r["w_mid_px"] = float(widths_px[1]); r["w_bot_px"] = float(widths_px[2])
                r["w_top_mm"] = float(widths_mm[0]); r["w_mid_mm"] = float(widths_mm[1]); r["w_bot_mm"] = float(widths_mm[2])

                (ax1,ay1,ax2,ay2) = wsegs_px[0]
                (bx1_,by1_,bx2_,by2_) = wsegs_px[1]
                (cx1,cy1,cx2,cy2) = wsegs_px[2]

                r["w_top_x1_mm"] = float(ax1)*mmx; r["w_top_y1_mm"] = float(ay1)*local_mmy
                r["w_top_x2_mm"] = float(ax2)*mmx; r["w_top_y2_mm"] = float(ay2)*local_mmy

                r["w_mid_x1_mm"] = float(bx1_)*mmx; r["w_mid_y1_mm"] = float(by1_)*local_mmy
                r["w_mid_x2_mm"] = float(bx2_)*mmx; r["w_mid_y2_mm"] = float(by2_)*local_mmy

                r["w_bot_x1_mm"] = float(cx1)*mmx; r["w_bot_y1_mm"] = float(cy1)*local_mmy
                r["w_bot_x2_mm"] = float(cx2)*mmx; r["w_bot_y2_mm"] = float(cy2)*local_mmy

                r["beam_idx"] = -1
                r["zone_v"] = "NA"
                r["zone_h"] = "NA"

            crack_seg_rows.extend(seg_rows)
            crack_idx += 1

        _append_rows_or_zero(rows, rel, crack_seg_rows)

    # aplicar escala a dx/dy -> dx_mm/dy_mm para plots
    apply_mm_scale_to_rows(rows, scale_map)
    compute_local_angles_mm(rows)

    _write_lengths_csv(out_csv_path, rows)

    # plots inf (si run_dir se pasó)
    if run_dir is not None:
        process_crack_segments_mm(rows, Path(run_dir), "plots_mm_inf")

    return out_csv_path
# =========================
# Tidy / Prune
# =========================
def tidy_run_layout(run_project, run_dir, resolved_run_name):
    def _move_latest(prefix, subfolder):
        pattern = os.path.join(run_project, prefix + "*")
        cands = [d for d in glob(pattern) if os.path.isdir(d)]
        if not cands: return
        src = max(cands, key=os.path.getmtime)
        dst = os.path.join(run_dir, subfolder)
        if os.path.isdir(dst): shutil.rmtree(dst)
        shutil.move(src, dst)
        print(f"[Tidy] {os.path.basename(src)} -> {dst}")
    _move_latest(f"{resolved_run_name}_val_final", "val")
    _move_latest(f"{resolved_run_name}_val_pred",  "pred_val")
    _move_latest(f"{resolved_run_name}_test_pred", "pred_test")
    _move_latest(f"{resolved_run_name}_new",       "inference")

def prune_old_runs(run_project, resolved_run_name, run_dir, keep_last=2):
    base = run_project
    if not os.path.isdir(base): return
    prefix = re.sub(r"\d+$", "", resolved_run_name)
    dirs = [d for d in glob(os.path.join(base, f"{prefix}*")) if os.path.isdir(d)]
    dirs = [d for d in dirs if os.path.abspath(d) != os.path.abspath(run_dir)]
    if not dirs:
        print(f"[Prune] No hay runs obsoletos para borrar (prefijo '{prefix}')."); return
    dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    keep = dirs[:keep_last]; trash = dirs[keep_last:]
    for d in trash:
        try: shutil.rmtree(d); print(f"[Prune] Eliminado: {d}")
        except Exception as e: print(f"[Prune] Error al eliminar {d}: {e}")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="YOLOv11 SEG (Fisura/Apoyo)")
    # Dataset maestros
    ap.add_argument("--base",   default=r"C:\dev\data\beam_seg")
    ap.add_argument("--images", default="all_images")   # FOTOS
    ap.add_argument("--labels", default="all_labels")   # POLÍGONOS Fisura/Apoyo
    ap.add_argument("--labels_viga", default="labels")  # Viga (para clasificación val/test)
    ap.add_argument("--ds_name", default="seg_ds")      # dataset de trabajo

    # TAKE / split
    ap.add_argument("--limit",  type=int, default=None, help="Si no se especifica, se preguntará por consola.")
    ap.add_argument("--ptrain", type=float, default=0.8)
    ap.add_argument("--pval",   type=float, default=0.1)
    ap.add_argument("--ptest",  type=float, default=0.1)
    ap.add_argument("--names",  default="Apoyo,Fisura")
    ap.add_argument("--seed",   type=int, default=42)

    # Negativos opcionales
    ap.add_argument("--add_negatives", action="store_true")
    ap.add_argument("--neg_count", type=int, default=None)

    # Entrenamiento
    ap.add_argument("--model",     default="yolo11s-seg.pt")
    ap.add_argument("--epochs",    type=int, default=50)
    ap.add_argument("--batch",     type=int, default=4)
    ap.add_argument("--imgsz",     type=int, default=960)
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--workers",   type=int, default=0)
    ap.add_argument("--patience",  type=int, default=10)
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0",       type=float, default=0.005)
    ap.add_argument("--lrf",       type=float, default=0.01)

    # Augmentations
    ap.add_argument("--mosaic",    type=float, default=0.2)
    ap.add_argument("--mixup",     type=float, default=0.10)
    ap.add_argument("--copy_paste",type=float, default=0.20)
    ap.add_argument("--hsv_s",     type=float, default=0.7)
    ap.add_argument("--scale",     type=float, default=0.5)
    ap.add_argument("--fliplr",    type=float, default=0.5)
    ap.add_argument("--flipud",    type=float, default=0.1)
    ap.add_argument("--degrees",   type=float, default=15.0)
    ap.add_argument("--translate", type=float, default=0.10)
    ap.add_argument("--shear",     type=float, default=2.0)
    ap.add_argument("--perspective", type=float, default=0.001)
    ap.add_argument("--auto_augment", default="randaugment")
    ap.add_argument("--erasing",   type=float, default=0.4)

    ap.add_argument("--plots", dest="plots", action="store_true")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.set_defaults(plots=True)

    # Runs e inferencia
    ap.add_argument("--run_project", default=r"C:\dev\PYTHON WORK\runs_seg")
    ap.add_argument("--run_name",    default="beam_seg")
    ap.add_argument("--prune_keep_last", type=int, default=2)
    # Escala anisotrópica (CSV proveniente de pose_fis_col.py)
    ap.add_argument("--scale_csv", default=None,
                    help="CSV con columnas image, mm_per_px_x, mm_per_px_y (exportado por dect_fis_col.py -> beam_scale_viga.csv)")

    # Inferencia: pedir N imágenes sin label
    ap.add_argument("--infer_n", type=int, default=None, help="Número de imágenes sin label para inferencia (0 omite).")
    ap.add_argument(
        "--split_from",
        default=r"C:\dev\data\beam_yolo",
        help="Ruta donde dect_fis_col.py guardó split_manifest.json o split_*.txt"
    )
    args = ap.parse_args()

    # Blindaje explícito de dispositivo para las fases de predicción posteriores al entrenamiento.
    # En CPU debe quedar como "cpu"; en GPU puede pasarse, por ejemplo, "0".
    print(f"[DEVICE] Ultralytics usará device={args.device}")

    base     = args.base
    img_dir  = os.path.join(base, args.images)
    lbl_dir  = os.path.join(base, args.labels)        # Fisura/Apoyo (polígonos)
    viga_dir = os.path.join(base, args.labels_viga)   # Viga (bb/seg) — solo para clasificar
    ds_root  = os.path.join(base, args.ds_name)

    # =========================
    # Escala anisotrópica (mm/px) por imagen
    # =========================
    # 1) Si NO te pasan --scale_csv, buscamos automáticamente el último CSV de DECT
    #    generado por dect_fis_col.py:
    #    runs_det\<run>\beam_scale\beam_scale_viga.csv
    if not args.scale_csv:
        default_det_runs = r"C:\dev\PYTHON WORK\runs_det"
        pattern = os.path.join(default_det_runs, "*", "beam_scale", "beam_scale_viga.csv")
        cands = glob(pattern)
        if cands:
            args.scale_csv = max(cands, key=os.path.getmtime)
            print(f"[Scale] Usando CSV de escala (DECT) por defecto: {args.scale_csv}")
        else:
            print("[Scale] No se encontró beam_scale_viga.csv en runs_det; se usará escala 1 mm/px por defecto.")

    # 2) Cargamos el mapa de escala (si el CSV existe). Para cualquier imagen
    #    que no aparezca en el mapa, luego se usará (1.0, 1.0).
    scale_map, profile_map = load_scale_resources(args.scale_csv) if args.scale_csv else ({}, {})

    # Estructura y limpieza del dataset de trabajo
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(ds_root, exist_ok=True)
    ensure_split_dirs(ds_root)
    clean_split_dirs(ds_root)
    print("[Split] Limpieza hecha en seg_ds: images/{train,val,test} y labels/{train,val,test} vaciadas.")

    print("Normalizando extensiones de imagen en all_images…")
    print("Extensiones normalizadas:", normalize_img_exts(img_dir))

    if args.add_negatives:
        create_empty_labels_from_missing(img_dir, lbl_dir, max_count=args.neg_count, seed=args.seed)

    if audit_and_fix_labels(img_dir, lbl_dir) == 0:
        print("No hay pares válidos (imagen+label)."); sys.exit(1)

# =========================
# Usar splits ya creados por dect_fis_col.py
# =========================

    def _stems_from_folder(folder):
        stems=set()
        for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
            for p in glob(os.path.join(folder, e)):
                stems.add(Path(p).stem)
        return sorted(stems)

    train = _stems_from_folder(os.path.join(base, "images", "train"))
    val   = _stems_from_folder(os.path.join(base, "images", "val"))
    test  = _stems_from_folder(os.path.join(base, "images", "test"))
    
    print("\n[SPLIT] Usando carpetas heredadas desde beam_seg/images")
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")

# =========================
# FILTRO: solo usar stems con label de segmentación (beam_seg/all_labels)
# =========================
    seg_labs = label_stems(lbl_dir)  # lbl_dir = C:\dev\data\beam_seg\all_labels

    def _filter(stems, name):
        kept = [s for s in stems if s in seg_labs]
        dropped = len(stems) - len(kept)
        if dropped:
            print(f"[Split] {name}: descartados {dropped} sin label de SEG (aún no etiquetados).")
        return kept

    train = _filter(train, "train")
    val   = _filter(val,   "val")
    test  = _filter(test,  "test")

    print(f"[Split] Después de filtrar por SEG labels -> Train={len(train)} Val={len(val)} Test={len(test)}")

    if (len(train) + len(val) + len(test)) == 0:
        print("[Split] ERROR: no hay ningún par imagen+label de segmentación para entrenar.")
        sys.exit(1)

    det_lbl_train = os.path.join(args.split_from, "labels", "train")
    det_lbl_val   = os.path.join(args.split_from, "labels", "val")
    det_lbl_test  = os.path.join(args.split_from, "labels", "test")

    for stem in train:
        copy_pair_seg(stem, "train", img_dir, lbl_dir, ds_root)
        copy_det_label_if_exists(stem, "train", det_lbl_train, ds_root)

    for stem in val:
        copy_pair_seg(stem, "val", img_dir, lbl_dir, ds_root)
        copy_det_label_if_exists(stem, "val", det_lbl_val, ds_root)

    for stem in test:
        copy_pair_seg(stem, "test", img_dir, lbl_dir, ds_root)
        copy_det_label_if_exists(stem, "test", det_lbl_test, ds_root)
    # Verificación dura
    verify_split_integrity(ds_root, strict=True)

    # YAML (nombres Fisura, Apoyo)
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    yaml_path = write_yaml(ds_root, names)

    print("\n==== RESUMEN SPLIT ====")
    total_pairs = len(train) + len(val) + len(test)

    print(f"Pares totales usados (heredado de detección): {total_pairs}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("seg_data.yaml:", yaml_path)

    # === Inferencia: pedir N imágenes sin label (desde all_images/all_labels)
    missing_total = len(image_stems(img_dir) - label_stems(lbl_dir))
    if args.infer_n is None:
        try:
            sugerido = min(100, missing_total)
            raw = input(f"Hay {missing_total} imágenes SIN label. ¿Cuántas quieres para INFERENCIA? (0 omite, sugerido {sugerido}): ")
            infer_n = int(raw)
        except Exception:
            infer_n = 0
    else:
        infer_n = args.infer_n

    infer_dir = os.path.join(base, "inference")
    def _copy_if_ok(src, dst):
        ok=True
        try:
            with Image.open(src) as im: im.verify()
        except Exception: ok=False
        if ok: shutil.copy(src, dst)
        return ok

    if infer_n>0 and missing_total>0:
        os.makedirs(infer_dir, exist_ok=True)
        for f in os.listdir(infer_dir):
            fp=os.path.join(infer_dir,f)
            if os.path.isfile(fp): os.remove(fp)
        missing = sorted(list(image_stems(img_dir) - label_stems(lbl_dir)))
        random.seed(args.seed); random.shuffle(missing)
        copied=0; skipped=0
        for stem in missing[:infer_n]:
            src = get_image_path_by_stem(stem, img_dir)
            if src:
                if _copy_if_ok(src, os.path.join(infer_dir, os.path.basename(src))): copied+=1
                else: skipped+=1
        print(f"[Inferencia] Copiadas {copied} (omitidas {skipped}) a: {infer_dir}")
        predict_source = infer_dir if copied>0 else None
    else:
        predict_source = None
        print("[Inferencia] Omitida.")

    # Entrenar + Val + Pred + Infer (visuales)
    if not safe_import_ultralytics(): sys.exit(1)
    best_pt, run_dir = train_val_predict_all(
        yaml_path=yaml_path,
        run_project=args.run_project, run_name=args.run_name,
        device=args.device, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, workers=args.workers,
        model_name=args.model, patience=args.patience, optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
        mosaic=args.mosaic, mixup=args.mixup, copy_paste=args.copy_paste,
        hsv_s=args.hsv_s, scale=args.scale, fliplr=args.fliplr, plots=args.plots,
        degrees=args.degrees, translate=args.translate, shear=args.shear, perspective=args.perspective,
        auto_augment=args.auto_augment, erasing=args.erasing, flipud=args.flipud,
        predict_source=predict_source
    )

    # Clasificar Viga Fisurada/No Fisurada en val/test + CSV longitudes
    has_det = any(glob(os.path.join(ds_root, "labels_det", "val", "*.txt"))) or \
        any(glob(os.path.join(ds_root, "labels_det", "test", "*.txt")))

    if has_det:
        from ultralytics import YOLO
        seg_model = YOLO(best_pt)
        try:
            seg_model.to(args.device)
        except Exception:
            pass
        classify_val_test_with_labels(run_dir, ds_root, seg_model, conf_seg=0.25, scale_map=scale_map, profile_map=profile_map, device=args.device)
    else:
        print("[CLASS] No hay labels de viga para val/test en seg_ds; omitiendo clasificación.")

    # CSV de INFERENCIA: generar en run_dir/beam_class/crack_lengths_inf.csv (junto a val/test)
    if predict_source:
        from ultralytics import YOLO
        seg_model = YOLO(best_pt)
        try:
            seg_model.to(args.device)
        except Exception:
            pass
        beam_class_dir = os.path.join(run_dir, "beam_class")
        os.makedirs(beam_class_dir, exist_ok=True)
        infer_csv_final = os.path.join(beam_class_dir, "crack_lengths_inf.csv")
        try:
            export_infer_lengths_csv(seg_model, predict_source, infer_csv_final, conf_seg=0.25, scale_map=scale_map, profile_map=profile_map, run_dir=run_dir, device=args.device)
            print("[Infer CSV] Generado:", infer_csv_final)
        except Exception as e:
            print("[Infer CSV] Aviso al generar crack_lengths_inf.csv:", e)

    # Orden y limpieza
    run_name_resolved = os.path.basename(run_dir)
    tidy_run_layout(args.run_project, run_dir, run_name_resolved)
    prune_old_runs(args.run_project, run_name_resolved, run_dir, keep_last=args.prune_keep_last)

    open_folder(run_dir, "Carpeta del run")
    print("\n=== FIN ===")
    print("Mejor modelo:", best_pt)
    print("Run dir (ordenado):", run_dir)

if __name__ == "__main__":
    main()