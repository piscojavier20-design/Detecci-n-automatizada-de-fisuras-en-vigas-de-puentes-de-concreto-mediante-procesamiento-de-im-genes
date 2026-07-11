# dect_fis_col.py — End-to-end (SEGMENTACIÓN) con augmentations y utilidades
# audit -> split -> YAML -> train(seg) -> val(confusion) -> predict(val/test)
# -> inference(N sin labels) -> preview(idx) -> tidy -> prune

import os, re, shutil, random, argparse, sys, csv, subprocess, math
from glob import glob

# === Plots
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

# =========================
# Utilidades básicas
# =========================

from pathlib import Path
import json

def save_split_manifest(base_dir: Path, train, val, test, meta: dict) -> None:
    (base_dir / "split_train.txt").write_text("\n".join(train), encoding="utf-8")
    (base_dir / "split_val.txt").write_text("\n".join(val), encoding="utf-8")
    (base_dir / "split_test.txt").write_text("\n".join(test), encoding="utf-8")
    (base_dir / "split_manifest.json").write_text(
        json.dumps({"meta": meta, "train": train, "val": val, "test": test}, indent=2),
        encoding="utf-8"
    )

def normalize_img_exts(img_dir):
    renames = 0
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
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
        for p in glob(os.path.join(img_dir, ext)):
            stems.add(os.path.splitext(os.path.basename(p))[0])
    return stems

def label_stems(lbl_dir):
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

def open_image(img_path, label):
    ok = open_path(img_path)
    print((f"{label} abierto: " if ok else f"{label} guardado en: ") + img_path)
    return ok

def open_folder(folder_path, label="Carpeta"):
    ok = open_path(folder_path)
    print((f"{label} abierto: " if ok else f"{label} en: ") + folder_path)
    return ok

def show_plot(path, title):
    if not os.path.exists(path):
        print(f"[Plot] No existe: {path}")
        return False
    try:
        img = Image.open(path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"[Plot] No se pudo mostrar con Matplotlib ({e}). Abriendo con visor del SO…")
        return open_image(path, title)

# =========================
# Auditoría y negativos
# =========================
def audit_and_fix_labels(img_dir, lbl_dir):
    imgs = image_stems(img_dir)
    labels = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]
    fixed = collisions = 0; nomatch = []
    for lf in labels:
        src = os.path.join(lbl_dir, lf); stem = os.path.splitext(lf)[0]
        if stem in imgs: continue
        cand = strip_prefix(stem)
        if cand in imgs:
            dst = os.path.join(lbl_dir, cand + ".txt")
            if os.path.exists(dst): collisions += 1
            else: os.rename(src, dst); fixed += 1
        else:
            nomatch.append(lf)
    remaining = []
    for lf in nomatch:
        stem = os.path.splitext(lf)[0]
        split_try = re.sub(r"^[^_-]+[_-]", "", stem)
        candidates = {s for s in imgs if (s==stem or s==split_try or s in stem or stem in s or s.endswith(split_try))}
        if len(candidates) == 1:
            c = list(candidates)[0]
            src = os.path.join(lbl_dir, lf); dst = os.path.join(lbl_dir, c + ".txt")
            if not os.path.exists(dst): os.rename(src, dst); fixed += 1
            else: collisions += 1
        else:
            remaining.append(lf)
    imgs2 = image_stems(img_dir); labs2 = label_stems(lbl_dir); pairs = len(imgs2 & labs2)
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
# Split y YAML
# =========================
def collect_pairs(img_dir, lbl_dir): 
    return sorted(image_stems(img_dir) & label_stems(lbl_dir))

def ensure_split_dirs(base):
    for p in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
        os.makedirs(os.path.join(base, p), exist_ok=True)

def clean_split_dirs(base):
    for d in [os.path.join(base, "images", s) for s in ["train","val","test"]] + \
             [os.path.join(base, "labels", s) for s in ["train","val","test"]]:
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp): os.remove(fp)

def copy_pair(stem, split, img_dir, lbl_dir, base):
    src_img = get_image_path_by_stem(stem, img_dir)
    if src_img:
        shutil.copy(src_img, os.path.join(base, "images", split, os.path.basename(src_img)))
    lsrc = os.path.join(lbl_dir, stem+".txt")
    shutil.copy(lsrc, os.path.join(base, "labels", split, stem+".txt"))

def smart_split(n, p_train=0.8, p_val=0.1, p_test=0.1, *, seed=None):
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        rng = random.Random(seed)
        return (1, 1, 0) if rng.random() < 0.5 else (1, 0, 1)

    rng = random.Random(seed)
    a = rng.randint(1, n - 2)         # asegura val>=1 y test>=1
    b = rng.randint(a + 1, n - 1)
    t = a
    v = b - a
    s = n - b
    return (t, v, s)


def write_yaml(base, names):
    yaml_path = os.path.join(base, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: " + base.replace('\\','/') + "\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for i,n in enumerate(names): f.write(f"  {i}: {n}\n")
    return yaml_path

# === Verificación de integridad del split (sin solapamientos ni faltantes)
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
# Inferencia (set sin labels)
# =========================
def list_images(dirpath):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp")
    files = []
    for e in exts: files += glob(os.path.join(dirpath, e))
    return sorted(files)

def _is_decodable(img_path):
    try:
        with Image.open(img_path) as im:
            im.verify()
        return True
    except Exception:
        return False

def _copy_if_ok(src, dst):
    ok = True
    if os.path.splitext(src)[1].lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}:
        ok = _is_decodable(src)
    if ok:
        shutil.copy(src, dst)
    return ok

def make_inference_set_by_count(img_dir, lbl_dir, out_dir, count, seed=42, clean=False, shuffle=True):
    stems_all = image_stems(img_dir); stems_lab = label_stems(lbl_dir)
    missing_stems = list(stems_all - stems_lab)
    total_missing = len(missing_stems)
    if not total_missing:
        print("[Inferencia] No hay imágenes sin label en all_images."); return 0, 0
    if shuffle:
        random.seed(seed); random.shuffle(missing_stems)
    n = max(0, min(int(count), total_missing))
    if n == 0:
        print("[Inferencia] Se pidió 0 imágenes. Omitiendo."); return 0, total_missing

    os.makedirs(out_dir, exist_ok=True)
    if clean:
        for f in os.listdir(out_dir):
            fp = os.path.join(out_dir, f)
            if os.path.isfile(fp): os.remove(fp)

    copied = 0; skipped = 0
    for stem in missing_stems[:n]:
        src = get_image_path_by_stem(stem, img_dir)
        if src:
            ok = _copy_if_ok(src, os.path.join(out_dir, os.path.basename(src)))
            if ok: copied += 1
            else: skipped += 1
    print(f"[Inferencia] Copiadas {copied} (omitidas {skipped} corruptas) a: {out_dir}")
    return copied, total_missing

# =========================
# Entrenamiento / Métricas
# =========================
def safe_import_ultralytics():
    try:
        from ultralytics import YOLO  # noqa: F401
        return True
    except Exception as e:
        print("\n[ERROR] No se pudo importar 'ultralytics'. Instala con:  pip install ultralytics")
        print("Detalle:", e)
        return False

def summarize_csv(csv_path):
    if not os.path.exists(csv_path):
        print("No se encontró results.csv"); return []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("results.csv vacío"); return []
    last = rows[-1]

    def g(k, alt=None):
        try:
            return float(last.get(k, last.get(alt))) if (k in last or (alt and alt in last)) else float("nan")
        except: return float("nan")

    # Estos nombres funcionan para detect y segment (Ultralytics usa las mismas claves base)
    precision = g("metrics/precision(B)", "metrics/precision")
    recall    = g("metrics/recall(B)",    "metrics/recall")
    m50       = g("metrics/mAP50(B)",     "metrics/mAP50")
    m5095     = g("metrics/mAP50-95(B)",  "metrics/mAP50-95")

    # Pérdidas (segment agrega seg_loss además de box/cls/dfl)
    t_box     = g("train/box_loss"); t_cls = g("train/cls_loss"); t_dfl = g("train/dfl_loss"); t_seg = g("train/seg_loss")
    v_box     = g("val/box_loss");   v_cls = g("val/cls_loss");   v_dfl = g("val/dfl_loss");   v_seg = g("val/seg_loss")

    print("\n=== RESUMEN MÉTRICAS (último epoch) ===")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | mAP50: {m50:.3f} | mAP50-95: {m5095:.3f}")
    if (t_seg == t_seg) or (v_seg == v_seg):
        print(f"Train loss (box/cls/dfl/seg): {t_box:.3f}/{t_cls:.3f}/{t_dfl:.3f}/{t_seg:.3f}")
        print(f"Val   loss (box/cls/dfl/seg): {v_box:.3f}/{v_cls:.3f}/{v_dfl:.3f}/{v_seg:.3f}")
    else:
        print(f"Train loss (box/cls/dfl): {t_box:.3f}/{t_cls:.3f}/{t_dfl:.3f}")
        print(f"Val   loss (box/cls/dfl): {v_box:.3f}/{v_cls:.3f}/{v_dfl:.3f}")
    return rows

def overfitting_diagnosis(rows):
    if not rows or len(rows) < 4:
        print("\n[Diag] Muy pocos epochs para evaluar sobreajuste."); return
    def s(key):
        vals = []
        for r in rows:
            try: vals.append(float(r.get(key, "nan")))
            except: vals.append(float("nan"))
        return [v for v in vals if v == v]
    tbox, vbox = s("train/box_loss"), s("val/box_loss")
    tcls, vcls = s("train/cls_loss"), s("val/cls_loss")
    map95 = s("metrics/mAP50-95(B)") or s("metrics/mAP50-95")
    flag_loss = (len(tbox)>3 and len(vbox)>3 and tbox[-1] < min(tbox[:-1]) and vbox[-1] > min(vbox[:-1]) + 0.03) or \
                (len(tcls)>3 and len(vcls)>3 and tcls[-1] < min(tcls[:-1]) and vcls[-1] > min(vcls[:-1]) + 0.03)
    flag_map = False
    if len(map95) > 3:
        peak = max(map95[:-1]); drop = peak - map95[-1]; flag_map = drop > 0.05
    if flag_loss or flag_map:
        print("\n[Diag] **Posible sobreajuste**.")
        if flag_loss: print("- Pérdida de entrenamiento ↓ mientras la de validación ↑.")
        if flag_map:  print(f"- mAP50-95 bajó ~{drop:.3f} desde su pico.")
    else:
        print("\n[Diag] No se observan signos claros de sobreajuste.")

# =========================
# Detección rápida del tipo de label
# =========================
def guess_label_kind(lbl_split_dir):
    txts = glob(os.path.join(lbl_split_dir, "*.txt"))
    for p in txts:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    parts=line.split()
                    if len(parts) < 5:  # inválido o vacío
                        continue
                    # ignora clase
                    coords = parts[1:]
                    # todos números?
                    try: _ = [float(x) for x in coords]
                    except: continue
                    if len(coords) == 4:
                        return "detect"
                    if len(coords) >= 6 and len(coords) % 2 == 0:
                        return "segment"
        except: 
            continue
    return "unknown"

# =========================
# Train + Val + Pred + Inference + Preview
# =========================
def train_val_predict_all(yaml_path, run_project, run_name, device, epochs, batch, imgsz, workers,
                          model_name, patience, optimizer, lr0, lrf,
                          mosaic, mixup, copy_paste, hsv_s, scale, fliplr, plots,
                          degrees, translate, shear, perspective, auto_augment, erasing, flipud,
                          predict_source):
    from ultralytics import YOLO

    print("\n=== ENTRENANDO YOLOv11 (segment) ===")
    model = YOLO(model_name)
    r = model.train(
        data=yaml_path, imgsz=imgsz, epochs=epochs, batch=batch, device=device, workers=workers,
        patience=patience, project=run_project, name=run_name,
        optimizer=optimizer, lr0=lr0, lrf=lrf,
        # Augmentations
        mosaic=mosaic, mixup=mixup, copy_paste=copy_paste,
        hsv_s=hsv_s, scale=scale, fliplr=fliplr, flipud=flipud,
        degrees=degrees, translate=translate, shear=shear, perspective=perspective,
        auto_augment=auto_augment, erasing=erasing,
        plots=plots
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
            best_pt = max(cands, key=os.path.getmtime)
            save_dir = os.path.dirname(os.path.dirname(best_pt))

    run_dir = save_dir
    results_png = os.path.join(run_dir, "results.png")
    results_csv = os.path.join(run_dir, "results.csv")
    print(f"[Run] Carpeta del entrenamiento: {run_dir}")
    print(f"[Run] Pesos: {best_pt}")

    resolved_run_name = os.path.basename(run_dir)

    # Validación (confusion + other plots)
    best_model = YOLO(best_pt)
    best_model.val(data=yaml_path, imgsz=imgsz, device=device, plots=True,
                   project=run_project, name=f"{resolved_run_name}_val_final")

    # Predicciones val/test (visualiza máscaras automáticamente)
    base = os.path.dirname(os.path.dirname(yaml_path))
    for split, outdir_name in [("val", f"{resolved_run_name}_val_pred"),
                               ("test", f"{resolved_run_name}_test_pred")]:
        src = os.path.join(base, "images", split)
        if os.path.isdir(src):
            best_model.predict(source=src, imgsz=imgsz, device=device,
                               project=run_project, name=outdir_name,
                               save=True, verbose=False)
            print(f"Predicciones {split} ->", os.path.join(run_project, outdir_name))

    # Predicciones NUEVAS (inferencia) + preview por índice
    infer_pred_dir = None
    if predict_source and os.path.isdir(predict_source):
        infer_name = f"{resolved_run_name}_new"
        print("\n=== PREDICCIONES NUEVAS (inferencia) ===")
        preds = best_model.predict(source=predict_source, imgsz=imgsz, device=device,
                                   project=run_project, name=infer_name, save=True, verbose=False)
        infer_pred_dir = os.path.join(run_project, infer_name)
        print("Inferencia guardada en:", infer_pred_dir)
        try:
            raw = input("¿Qué índice de imagen quieres mostrar? (ej. 49 para la #50): ")
            idx = int(raw)
        except Exception:
            idx = 0
        idx = max(0, min(len(preds)-1, idx))
        try:
            preds[idx].show()  # debe mostrar máscaras
            print(f"Mostrando predicción índice {idx}.")
        except Exception as e:
            print("No se pudo abrir .show(). Detalle:", e)
            imgs = sorted(glob(os.path.join(infer_pred_dir, "*.jpg")) + glob(os.path.join(infer_pred_dir, "*.png")))
            if imgs:
                open_image(imgs[min(idx, len(imgs)-1)], "Preview inferencia")
    else:
        print("\n(No se proporcionó carpeta de inferencia o no existe).")

    # Plots
    _ = show_plot(results_png, "Resultados de entrenamiento (losses y métricas)")
    cm_dirs = [d for d in glob(os.path.join(run_project, f"{resolved_run_name}_val_final*")) if os.path.isdir(d)]
    mostradas = 0
    for cm_dir in cm_dirs:
        for cm in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
            cm_path = os.path.join(cm_dir, cm)
            if os.path.exists(cm_path):
                if show_plot(cm_path, f"Matriz de confusión — {os.path.basename(cm_dir)}"):
                    mostradas += 1
    if not mostradas:
        print("[Plot] No encontré matrices de confusión para mostrar.")

    rows = summarize_csv(results_csv)
    overfitting_diagnosis(rows)

    print("\nPesos finales:", best_pt)
    return best_pt, run_dir, resolved_run_name

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

def prune_old_runs(run_project, resolved_run_name, run_dir, keep_last=1):
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
# Escala (labels_scale) + geometría de VIGA (L/H)
# =========================

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_or_create_scale_table(stems, labels_scale_dir: str) -> dict:
    """
    Crea/carga: labels_scale/scale_lengths.csv
    Columnas esperadas:
      image, length_distx_mm, length_disty_mm

    Convención:
      - length_distx_mm = LONGITUD real de la VIGA (mm)
      - length_disty_mm = ALTURA real de la VIGA (mm)

    Si no existe, lo crea con 1.0 y 1.0 para cada imagen.
    """
    ensure_dir(labels_scale_dir)
    csv_path = os.path.join(labels_scale_dir, "scale_lengths.csv")
    scale = {}

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = (row.get("image") or "").strip()
                if not img_name:
                    continue
                stem = Path(img_name).stem
                try:
                    dx_mm = float(row.get("length_distx_mm", "1") or "1")
                    dy_mm = float(row.get("length_disty_mm", "1") or "1")
                except ValueError:
                    dx_mm = dy_mm = 1.0
                scale[stem] = {"length_distx_mm": dx_mm, "length_disty_mm": dy_mm}

    changed = False
    for s in stems:
        if s not in scale:
            scale[s] = {"length_distx_mm": 1.0, "length_disty_mm": 1.0}
            changed = True

    if (not os.path.exists(csv_path)) or changed:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "length_distx_mm", "length_disty_mm"])
            for s in sorted(scale.keys()):
                w.writerow([f"{s}.jpg", scale[s]["length_distx_mm"], scale[s]["length_disty_mm"]])
        print(f"[SCALE] scale_lengths.csv listo/actualizado en: {csv_path}")

    return scale

def _img_wh(stem: str, img_dir: str):
    p = get_image_path_by_stem(stem, img_dir)
    if not p:
        return None, None, None
    try:
        im = Image.open(p)
        w, h = im.size
        im.close()
        return p, w, h
    except Exception:
        return p, None, None

def _parse_yolo_seg_line_to_points_px(line: str, img_w: int, img_h: int):
    """
    YOLO-seg line:
      cls x1 y1 x2 y2 ... (coords normalizadas)
    Retorna: cls_id, Nx2 puntos en px (float)
    """
    parts = line.strip().split()
    if len(parts) < 1 + 6:  # mínimo cls + 3 puntos
        return None
    try:
        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
    except Exception:
        return None
    if len(coords) % 2 != 0:
        return None
    xs = np.array(coords[0::2], dtype=float) * float(img_w)
    ys = np.array(coords[1::2], dtype=float) * float(img_h)
    pts = np.stack([xs, ys], axis=1)
    return cls_id, pts

def _pca_major_minor_lengths(pts: np.ndarray) -> tuple[float, float]:
    """
    Dado un set de puntos (Nx2) en px, estima:
      - length_px (eje mayor)
      - height_px (eje menor)
    usando PCA + proyecciones.
    """
    if pts is None or len(pts) < 3:
        return 0.0, 0.0
    c = pts.mean(axis=0, keepdims=True)
    X = pts - c
    cov = (X.T @ X) / max(len(X) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)  # orden ascendente
    v_major = evecs[:, 1]
    v_minor = evecs[:, 0]
    proj_major = X @ v_major
    proj_minor = X @ v_minor
    length_px = float(proj_major.max() - proj_major.min())
    height_px = float(proj_minor.max() - proj_minor.min())
    return max(length_px, 0.0), max(height_px, 0.0)

def _mask_from_poly_pts(pts: np.ndarray, W: int, H: int):
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon([tuple(map(float, p)) for p in pts], outline=1, fill=1)
    return np.array(img, dtype=np.uint8)


def _skeletonize_mask(mask_bin):
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask_bin.astype(bool)).astype(np.uint8)
    except Exception:
        return None


def _poly_centerline_pts(pts: np.ndarray, W: int, H: int):
    mask = _mask_from_poly_pts(pts, W, H).astype(bool)
    sk = _skeletonize_mask(mask.astype(np.uint8))
    if sk is not None and sk.any():
        ys, xs = np.where(sk > 0)
        if len(xs) >= 2:
            pts_xy, _, _, _ = _skeleton_centerline(sk > 0)
            if pts_xy is not None and len(pts_xy) >= 2:
                return pts_xy, mask
    # fallback PCA
    c = pts.mean(axis=0, keepdims=True)
    X = pts - c
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    pc1 = VT[0]
    proj = X @ pc1
    p1 = c[0] + proj.min() * pc1
    p2 = c[0] + proj.max() * pc1
    return np.array([[p1[0], p1[1]], [p2[0], p2[1]]], dtype=float), mask


def _skeleton_centerline(sk_bin):
    # reusa idea de diámetro sobre skeleton 8-conectado
    ys, xs = np.where(sk_bin)
    if len(ys) == 0:
        return None, 0.0, None, None
    nei8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    H, W = sk_bin.shape
    def neigh(y,x):
        for dy,dx in nei8:
            ny,nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny,nx,(1.0 if (dy==0 or dx==0) else math.sqrt(2.0))
    import heapq
    def dijkstra(src):
        sy,sx = src
        dist={(sy,sx):0.0}; prev={(sy,sx):None}; pq=[(0.0,sy,sx)]
        far=(sy,sx); far_d=0.0
        while pq:
            d,y,x = heapq.heappop(pq)
            if d > dist[(y,x)] + 1e-9:
                continue
            if d > far_d:
                far_d=d; far=(y,x)
            for ny,nx,w in neigh(y,x):
                if not sk_bin[ny,nx]:
                    continue
                nd=d+w
                if (ny,nx) not in dist or nd < dist[(ny,nx)]:
                    dist[(ny,nx)] = nd
                    prev[(ny,nx)] = (y,x)
                    heapq.heappush(pq,(nd,ny,nx))
        return prev, far, far_d
    endpoints=[]
    for y,x in zip(ys,xs):
        deg=0
        for dy,dx in nei8:
            ny,nx=y+dy,x+dx
            if 0 <= ny < H and 0 <= nx < W and sk_bin[ny,nx]:
                deg += 1
        if deg == 1:
            endpoints.append((int(y),int(x)))
    src = endpoints[0] if endpoints else (int(ys[0]), int(xs[0]))
    _, a, _ = dijkstra(src)
    prev, b, L = dijkstra(a)
    path=[]; cur=b
    while cur is not None:
        y,x = cur
        path.append((float(x), float(y)))
        cur = prev.get(cur)
    path.reverse()
    return np.array(path, dtype=float), float(L), a, b


def _measure_widths_on_centerline(mask: np.ndarray, pts_xy: np.ndarray, stations, max_step: int = 700):
    if pts_xy is None or len(pts_xy) < 2:
        return [0.0 for _ in stations]
    dxy = np.diff(pts_xy, axis=0)
    seglen = np.hypot(dxy[:,0], dxy[:,1])
    total = float(seglen.sum())
    if total < 1e-9:
        return [0.0 for _ in stations]
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    Hh, Ww = mask.shape
    def inb(ix, iy):
        return 0 <= ix < Ww and 0 <= iy < Hh
    widths=[]
    for tf in stations:
        tf = float(min(1.0, max(0.0, tf)))
        target = tf * total
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = max(0, min(j, len(pts_xy)-2))
        t0, t1 = cum[j], cum[j+1]
        alpha = 0.0 if (t1 - t0) < 1e-9 else (target - t0) / (t1 - t0)
        p = (1-alpha)*pts_xy[j] + alpha*pts_xy[j+1]
        tang = pts_xy[j+1] - pts_xy[j]
        nrm = float(np.hypot(tang[0], tang[1]))
        tang = np.array([1.0, 0.0]) if nrm < 1e-9 else tang / nrm
        nx, ny = -float(tang[1]), float(tang[0])
        cx, cy = float(p[0]), float(p[1])
        ppx, ppy = cx, cy
        for _ in range(max_step):
            ix, iy = int(round(ppx)), int(round(ppy))
            if (not inb(ix, iy)) or (not mask[iy, ix]):
                break
            ppx += nx; ppy += ny
        pmx, pmy = cx, cy
        for _ in range(max_step):
            ix, iy = int(round(pmx)), int(round(pmy))
            if (not inb(ix, iy)) or (not mask[iy, ix]):
                break
            pmx -= nx; pmy -= ny
        widths.append(float(math.hypot(ppx-pmx, ppy-pmy)))
    return widths


def estimate_beam_local_profile_from_labels(stem: str, img_dir: str, lbl_dir: str, viga_class_id: int):
    img_path, w, h = _img_wh(stem, img_dir)
    if not w or not h:
        return 0.0, 0.0, []
    lp = os.path.join(lbl_dir, stem + ".txt")
    if not os.path.exists(lp):
        return 0.0, 0.0, []

    best = {"area": 0.0, "L": 0.0, "H": 0.0, "profile": []}
    with open(lp, "r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_yolo_seg_line_to_points_px(line, w, h)
            if not parsed:
                continue
            cls_id, pts = parsed
            if cls_id != viga_class_id:
                continue
            center_pts, mask = _poly_centerline_pts(pts, w, h)
            L, H = _pca_major_minor_lengths(pts)
            stations = [i/100.0 for i in range(101)]
            widths = _measure_widths_on_centerline(mask, center_pts, stations)
            area = float(max(L,0.0) * max(H,0.0))
            if area > best["area"]:
                best = {
                    "area": area,
                    "L": L,
                    "H": H,
                    "profile": [{"u_pct": round(s*100.0, 4), "local_height_px": round(float(wp), 4)} for s, wp in zip(stations, widths)]
                }
    return float(best["L"]), float(best["H"]), best["profile"]


def export_beam_scale_csv(run_dir: str, base: str, img_dir: str, lbl_dir: str,
                          labels_scale_dir: str, scale_table: dict, viga_class_id: int):
    """
    Genera:
      - run_dir/beam_scale/beam_scale_viga.csv   (escala global por imagen)
      - run_dir/beam_scale/beam_scale_profile.csv (perfil local u_pct -> mm/px_y_local)
    """
    out_dir = os.path.join(run_dir, "beam_scale")
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "beam_scale_viga.csv")
    out_profile_csv = os.path.join(out_dir, "beam_scale_profile.csv")

    rows = []
    profile_rows = []
    for subset in ("train", "val", "test"):
        split_dir = os.path.join(base, "images", subset)
        if not os.path.isdir(split_dir):
            continue
        stems = []
        for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
            stems += [Path(p).stem for p in glob(os.path.join(split_dir, e))]
        for stem in sorted(set(stems)):
            Lpx, Hpx, profile = estimate_beam_local_profile_from_labels(stem, img_dir, lbl_dir, viga_class_id=viga_class_id)

            Lmm = float(scale_table.get(stem, {}).get("length_distx_mm", 1.0))
            Hmm = float(scale_table.get(stem, {}).get("length_disty_mm", 1.0))

            mm_px_x = (Lmm / Lpx) if (Lpx and Lpx > 0) else 0.0
            mm_px_y = (Hmm / Hpx) if (Hpx and Hpx > 0) else 0.0
            vals = [v for v in (mm_px_x, mm_px_y) if v > 0]
            mm_px_mean = (sum(vals) / len(vals)) if vals else 0.0

            rows.append({
                "subset": subset,
                "image": f"{stem}.jpg",
                "length_px": round(Lpx, 4),
                "height_px": round(Hpx, 4),
                "length_mm": Lmm,
                "height_mm": Hmm,
                "mm_per_px_x": round(mm_px_x, 8),
                "mm_per_px_y": round(mm_px_y, 8),
                "mm_per_px_mean": round(mm_px_mean, 8),
            })
            for p in profile:
                local_h = float(p.get("local_height_px", 0.0))
                profile_rows.append({
                    "subset": subset,
                    "image": f"{stem}.jpg",
                    "u_pct": float(p.get("u_pct", 0.0)),
                    "local_height_px": round(local_h, 4),
                    "beam_height_mm": Hmm,
                    "mm_per_px_y_local": round((Hmm / local_h) if (Hmm > 0 and local_h > 1e-9) else 0.0, 8),
                })

    if not rows:
        print("[beam_scale] No se pudieron crear filas. ¿Hay labels de viga (YOLO-seg) y la clase correcta?")
        return

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[beam_scale] CSV global creado en: {out_csv}")

    if profile_rows:
        with open(out_profile_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
            w.writeheader()
            w.writerows(profile_rows)
        print(f"[beam_scale] Perfil local creado en: {out_profile_csv}")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="YOLO end-to-end (SEGMENTACIÓN) con augmentations y limpieza de runs")
    # Dataset
    ap.add_argument("--base",   default=r"C:\dev\data\beam_yolo")
    ap.add_argument("--images", default="all_images")
    ap.add_argument("--labels", default="all_labels")
    ap.add_argument("--limit",  type=int, default=None, help="Si no se especifica, se preguntará por consola.")
    ap.add_argument("--ptrain", type=float, default=0.8)
    ap.add_argument("--pval",   type=float, default=0.1)
    ap.add_argument("--ptest",  type=float, default=0.1)
    ap.add_argument("--names",  default="beam", help="Clases separadas por coma.")
    ap.add_argument("--seed",   type=int, default=42)

    # Negativos opcionales
    ap.add_argument("--add_negatives", action="store_true")
    ap.add_argument("--neg_count", type=int, default=None)

    # Tarea: detect o segment (por defecto segment)
    ap.add_argument("--task", choices=["segment", "detect"], default="segment")

    ap.add_argument("--data_dir", default="data",help="Subcarpeta dentro de --base para archivos auxiliares (por defecto: base/data).")
    ap.add_argument("--labels_scale_subdir", default=os.path.join("labels_scale"),help="Subcarpeta dentro de base/data donde vive scale_lengths.csv.")
    ap.add_argument("--viga_class_id", type=int, default=0,help="ID de clase (YOLO) correspondiente a VIGA dentro de tus labels (YOLO-seg).")

    # Entrenamiento & HP (incluye augmentations fuertes por defecto)
    ap.add_argument("--model",     default="yolo11s-seg.pt")
    ap.add_argument("--epochs",    type=int, default=30)
    ap.add_argument("--batch",     type=int, default=4)
    ap.add_argument("--imgsz",     type=int, default=960)
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--workers",   type=int, default=0)
    ap.add_argument("--patience",  type=int, default=10)
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0",       type=float, default=0.005)
    ap.add_argument("--lrf",       type=float, default=0.01)

    # Augmentations (por defecto activas y moderadas)
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
    ap.add_argument("--auto_augment", default="randaugment")  # 'randaugment' | 'augmix' | 'none'
    ap.add_argument("--erasing",   type=float, default=0.4)

    # Plots ON por defecto; se puede desactivar con --no-plots
    ap.add_argument("--plots", dest="plots", action="store_true")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.set_defaults(plots=True)

    # Runs e inferencia
    ap.add_argument("--run_project", default=r"C:\dev\PYTHON WORK\runs")
    ap.add_argument("--run_name",    default="beam_seg")

    # Prune
    ap.add_argument("--prune_keep_last", type=int, default=1)

    # Inferencia: pedir N imágenes
    ap.add_argument("--infer_n", type=int, default=None, help="Número de imágenes sin label para inferencia (si no, se pregunta).")

    args = ap.parse_args()

    base    = args.base
    img_dir = os.path.join(base, args.images)
    lbl_dir = os.path.join(base, args.labels)

    # =========================
    # labels_scale (base/data/labels_scale)
    # =========================
    data_dir = os.path.join(base, args.data_dir)
    labels_scale_dir = os.path.join(data_dir, args.labels_scale_subdir)

    # Vamos a crear/asegurar scale_lengths.csv para todas las imágenes existentes
    # (train/val/test se crean más adelante, pero aquí dejamos lista la tabla).
    all_stems = sorted(list(image_stems(img_dir)))
    scale_table = load_or_create_scale_table(all_stems, labels_scale_dir)

    # Estructura y limpieza
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    ensure_split_dirs(base)
    clean_split_dirs(base)
    print("[Split] Limpieza hecha: images/{train,val,test} y labels/{train,val,test} vaciadas.")

    print("Normalizando extensiones de imagen...")
    print("Extensiones normalizadas:", normalize_img_exts(img_dir))

    if args.add_negatives:
        create_empty_labels_from_missing(img_dir, lbl_dir, max_count=args.neg_count, seed=args.seed)

    if audit_and_fix_labels(img_dir, lbl_dir) == 0:
        print("No hay pares válidos (imagen+label)."); sys.exit(1)

    # TAKE + SHUFFLE — prompt si no se pasó --limit
    pairs = collect_pairs(img_dir, lbl_dir); total_pairs = len(pairs)
    if args.limit is None:
        try: limit = int(input(f"Hay {total_pairs} pares disponibles. ¿Cuántos quieres usar? "))
        except Exception: print("Entrada inválida."); sys.exit(1)
    else:
        limit = args.limit
    limit = max(1, min(limit, total_pairs))
    random.seed(args.seed); random.shuffle(pairs)
    subset = pairs[:limit]


# =========================
# SPLIT CONTROLADO POR TERMINAL (NO smart_split)
# =========================
    n = len(subset)
    print(f"\n[SPLIT] Vas a usar {n} imágenes (TAKE). Ahora define el split:")

    raw = input("Ingresa n_train n_val n_test (ej: 80 10 10) [Enter = 80/10/10]: ").strip()

    if not raw:
        n_train = int(round(0.8 * n))
        n_val   = int(round(0.1 * n))
        n_test  = n - n_train - n_val
    else:
        parts = raw.split()
        if len(parts) != 3:
            print("Entrada inválida. Deben ser 3 números. Ej: 80 10 10")
            sys.exit(1)
        n_train, n_val, n_test = map(int, parts)
        if n_train < 1 or n_val < 0 or n_test < 0:
            print("Entrada inválida: n_train>=1, n_val>=0, n_test>=0")
            sys.exit(1)
        if n_train + n_val + n_test != n:
            print(f"Entrada inválida: n_train+n_val+n_test debe ser {n} (tu TAKE).")
            sys.exit(1)

    train = subset[:n_train]
    val   = subset[n_train:n_train+n_val]
    test  = subset[n_train+n_val:n_train+n_val+n_test]

    print(f"[SPLIT] Train={len(train)}  Val={len(val)}  Test={len(test)}")

    meta = {
      "seed": args.seed,
        "limit": limit,
       "total_pairs": total_pairs,
        "ptrain": args.ptrain,
        "pval": args.pval,
        "ptest": args.ptest,
    }
    save_split_manifest(Path(base), train, val, test, meta)

    for stem in train: copy_pair(stem, "train", img_dir, lbl_dir, base)
    for stem in val:   copy_pair(stem, "val",   img_dir, lbl_dir, base)
    for stem in test:  copy_pair(stem, "test",  img_dir, lbl_dir, base)

    # Verificación dura
    verify_split_integrity(base, strict=True)

    # YAML
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    yaml_path = write_yaml(base, names)

    print("\n==== RESUMEN SPLIT ====")
    print(f"Pares totales: {total_pairs}  |  Usados (TAKE): {limit}  |  SHUFFLE: True")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("data.yaml:", yaml_path)

    # Chequeo del tipo de label según la tarea
    lbl_train_dir = os.path.join(base, "labels", "train")
    kind = guess_label_kind(lbl_train_dir)
    if args.task == "segment" and kind == "detect":
        print("\n[ERROR] Tus labels parecen de DETECCIÓN (cajas). Para SEGMENTACIÓN necesitas polígonos:")
        print("- Formato YOLO-seg: 'clase x1 y1 x2 y2 ...' con coords normalizadas [0,1].")
        print("- Convierte tus etiquetas (LabelMe/Roboflow/Ultralytics Label Studio) y vuelve a ejecutar.")
        sys.exit(1)
    if args.task == "detect" and kind == "segment":
        print("\n[Aviso] Etiquetas con polígonos detectadas pero --task=detect. Cambia a --task segment para usar máscaras.")

    # === Inferencia: pedir N imágenes sin label ===
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
    copied, _ = make_inference_set_by_count(img_dir, lbl_dir, infer_dir, count=infer_n,
                                            seed=args.seed, clean=True, shuffle=True)
    predict_source = infer_dir if (infer_n > 0 and copied > 0) else None

    # Entrenar + Val + Pred + Infer + Preview(índice)
    if not safe_import_ultralytics(): sys.exit(1)
    best_pt, run_dir, resolved_run_name = train_val_predict_all(
        yaml_path=yaml_path,
        run_project=args.run_project, run_name=args.run_name,
        device=args.device, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, workers=args.workers,
        model_name=args.model if args.task=="segment" else "yolo11s.pt",
        patience=args.patience, optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
        mosaic=args.mosaic, mixup=args.mixup, copy_paste=args.copy_paste,
        hsv_s=args.hsv_s, scale=args.scale, fliplr=args.fliplr, plots=args.plots,
        degrees=args.degrees, translate=args.translate, shear=args.shear, perspective=args.perspective,
        auto_augment=args.auto_augment, erasing=args.erasing, flipud=args.flipud,
        predict_source=predict_source
    )

    # =========================
    # Exportar escala mm/px usando labels_scale + geometría de VIGA desde labels
    # =========================
    export_beam_scale_csv(
        run_dir=run_dir,
        base=base,
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        labels_scale_dir=labels_scale_dir,
        scale_table=scale_table,
        viga_class_id=args.viga_class_id
    )

    # Ordenar y limpiar runs viejos
    tidy_run_layout(args.run_project, run_dir, resolved_run_name)
    prune_old_runs(args.run_project, resolved_run_name, run_dir, keep_last=args.prune_keep_last)

    # Abrir carpeta del run
    open_folder(run_dir, "Carpeta del run")
    val_dir = os.path.join(run_dir, "val")
    if os.path.isdir(val_dir): open_folder(val_dir, "Carpeta de validación (matrices)")

    print("\n=== FIN ===")
    print("Mejor modelo:", best_pt)
    print("Run dir (ordenado):", run_dir)

if __name__ == "__main__":
    main()
