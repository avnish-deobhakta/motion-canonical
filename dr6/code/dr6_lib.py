"""
dr6_lib.py  --  DR6: general-purpose multimodal models screening for diabetic
retinopathy across six public datasets (paper 2 of the Motion Canonical programme).

One shared library for every dr6 notebook.  Notebooks stay thin; everything that
affects the data lives here, is versioned in one place, and its sha256 is written
into every manifest.

Design locks (change only with a logged amendment):
  * Six datasets: APTOS 2019, EyePACS (Kaggle 2015), IDRiD, DDR, SUSTech-SYSU, BRSET.
  * Stratified sample: up to N_PER_GRADE gradable images per ICDR grade 0..4 per
    dataset, one image per patient where a patient id exists, fixed seed.
    Plus N_UNGRADABLE ungradable images from DDR (grade 5) and BRSET (quality
    "Inadequate").  The 200 APTOS images used in paper 1 are excluded.
  * Every image: crop to the fundus field of view, pad to square, Lanczos to
    1024 x 1024, JPEG q95 4:4:4, no metadata.  Native field-of-view diameter is
    recorded as a covariate.
  * BRSET (PhysioNet credentialed) never leaves our own compute: local open-weight
    model only, and nothing derived from it is sent to any third-party service
    unless a written PhysioNet approval is logged in config (BRSET_THIRD_PARTY_OK).
  * Vendor defaults for sampling and reasoning; only the output ceiling is set.
    Every call logs the served model string and token usage.  One shuffled queue
    across vendors and arms.
"""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import io
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LIB_VERSION = "dr6-lib-0.3.1"

# ---------------------------------------------------------------------------
# Paths (Colab Drive mount).  Dataset locations are checked by notebook 20.
# ---------------------------------------------------------------------------
MYDRIVE = Path("/content/drive/MyDrive")
ROOT = MYDRIVE / "retina_image_ai" / "dr6"
KEY_DIR = ROOT / "_KEY_DO_NOT_SHIP"          # ground truth; opened only after the scoring freeze
SETS = ROOT / "sets"                          # normalised images + registry (no labels)
RUNS = ROOT / "runs"
CONFIG = ROOT / "config"
PREREG = ROOT / "prereg"

DATA = MYDRIVE / "stochastic_orthogonality" / "data"
SOURCES: dict[str, dict] = {
    "aptos": {
        "country": "India", "license": "Kaggle competition rules (APTOS 2019)",
        "labels": DATA / "aptos" / "raw" / "train.csv",
        "images": DATA / "aptos" / "raw" / "train_images",
        "paper1_key": MYDRIVE / "retina_image_ai" / "runs" / "eyepacs_20260828T012620Z" / "_KEY_DO_NOT_SHIP"
                      / "key_eyepacs_eyepacs_20260828T012620Z.csv",
    },
    "eyepacs": {
        "country": "United States", "license": "Kaggle competition rules (Diabetic Retinopathy Detection 2015)",
        "labels": DATA / "eyepacs" / "raw" / "trainLabels.csv",
        "zip": DATA / "eyepacs" / "raw" / "train_combined.zip",
    },
    "idrid": {
        "country": "India", "license": "CC BY 4.0 (IEEE DataPort)",
        "root": MYDRIVE / "idrid",
    },
    "ddr": {
        "country": "China", "license": "DDR dataset terms (research use)",
        "root": MYDRIVE / "ddr_dataset",
    },
    "sustech": {
        "country": "China", "license": "CC0 (SUSTech-SYSU)",
        "gdrive_id": "1QKM05fYr_S5lh4v0r-jL7HLrxqphNpoe",
        "local": Path("/content/sustech_raw"),
    },
    "brset": {
        "country": "Brazil", "license": "PhysioNet Credentialed Health Data License 1.5.0",
        "labels": DATA / "brset" / "label_brset.csv",
        "tar": DATA / "brset" / "brset_1.0.2.tar",
        "restricted": True,
    },
}
DATASETS = list(SOURCES)
RESTRICTED = {k for k, v in SOURCES.items() if v.get("restricted")}

SEED = 20261016
N_PER_GRADE = 48
N_UNGRADABLE = {"ddr": 100, "brset": 100}
NORM_SIDE = 1024
NORM_JPEG_Q = 95

# ---------------------------------------------------------------------------
# Models and arms
# ---------------------------------------------------------------------------
API_VENDORS = ["anthropic", "openai", "google", "xai"]
LOCAL_VENDORS = ["qwen"]
ALL_IMAGE_MODELS = API_VENDORS + LOCAL_VENDORS

GRADES = ["0", "1", "2", "3", "4", "ungradable"]
ICDR_TEXT = {
    "0": "No apparent diabetic retinopathy.",
    "1": "Mild non-proliferative DR: microaneurysms only.",
    "2": "Moderate non-proliferative DR: more than microaneurysms only, but less than severe.",
    "3": "Severe non-proliferative DR: any of more than 20 intraretinal haemorrhages in each of 4 quadrants, "
         "definite venous beading in 2 or more quadrants, or prominent intraretinal microvascular abnormalities "
         "in 1 or more quadrants, and no signs of proliferative DR.",
    "4": "Proliferative DR: neovascularisation, or vitreous or preretinal haemorrhage.",
    "ungradable": "Image quality too poor to grade.",
}

PROMPTS = {
    # Paper 1 screening prompt, extended with an explicit ungradable option.
    "forced": (
        "You are assisting a diabetic retinopathy screening programme. This is a colour fundus "
        "photograph from a patient with diabetes. Grade it on the International Clinical Diabetic "
        "Retinopathy (ICDR) scale, where 0=no DR, 1=mild NPDR, 2=moderate NPDR, 3=severe NPDR, "
        "4=proliferative DR. If the image quality is too poor to grade, give \"ungradable\" as the "
        "grade. Referable means grade 2 or worse. Reply ONLY as JSON, no markdown fences: "
        '{"icdr_grade":0,"referable":false,"confidence":0.0}'
    ),
    # Findings description; deliberately no grading scale, so stage 2 has work to do.
    "free": (
        "You are assisting a retina specialist. This is a colour fundus photograph from a patient "
        "with diabetes. In plain prose, describe the image quality and every retinal finding you see, "
        "with its location and extent (for example microaneurysms, haemorrhages and in how many "
        "quadrants, venous beading, intraretinal microvascular abnormalities, new vessels, hard "
        "exudates, cotton wool spots, laser scars). Then give your overall impression in one "
        "sentence. Do not use a numbered grading scale."
    ),
}
K_REPEATS = {"forced": 3, "free": 1}

# Stage 2: free text -> ICDR grade
SELF_PASS_PROMPT = (
    "Below is a description of a colour fundus photograph from a patient with diabetes, written by a "
    "colleague who looked at the image. Using only this description, grade the eye on the International "
    "Clinical Diabetic Retinopathy (ICDR) scale, where 0=no DR, 1=mild NPDR, 2=moderate NPDR, 3=severe NPDR, "
    "4=proliferative DR, or \"ungradable\" if the description says the image cannot be assessed. "
    "Reply ONLY as JSON, no markdown fences: {\"icdr_grade\":0,\"confidence\":0.0}\n\nDescription:\n"
)
JEV_MODEL = "typesafe/jev-1.14"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_INSTRUCTIONS = (
    "The state is a clinician's written description of a colour fundus photograph from a patient with "
    "diabetes. Choose the International Clinical Diabetic Retinopathy (ICDR) grade that the description "
    "supports."
)

MAX_OUTPUT_TOKENS = 16000
HTTP_TIMEOUT_S = 300
MAX_ATTEMPTS = 6
SUCCESS_RATE_FLOOR = 0.98
DEFAULT_CONCURRENCY = {"anthropic": 6, "openai": 6, "google": 4, "xai": 4, "openrouter": 4}
EXTRA_BODY: dict[str, dict] = {}
KEY_ENV_NAMES = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "xai": ["XAI_API_KEY", "GROK_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lib_sha() -> str:
    return sha256_file(Path(__file__))


def write_json(p: Path, obj: Any) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(p)


def read_json(p: Path) -> Any:
    return json.loads(Path(p).read_text())


def get_api_key(vendor: str) -> str | None:
    names = KEY_ENV_NAMES[vendor]
    try:
        from google.colab import userdata  # type: ignore
        for n in names:
            try:
                v = userdata.get(n)
                if v:
                    return v
            except Exception:
                pass
    except Exception:
        pass
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return None


def new_image_id(rng: random.Random) -> str:
    return "%016x" % rng.getrandbits(64)


# ---------------------------------------------------------------------------
# Image normalisation
# ---------------------------------------------------------------------------
def fov_bbox(arr, thresh: int = 20):
    """Bounding box (y0, y1, x0, x1) of the largest bright connected region (the
    fundus field of view), found on a 4x reduced red channel."""
    import numpy as np
    from scipy import ndimage as ndi
    small = arr[::4, ::4, 0]
    lab, n = ndi.label(small > thresh)
    h, w = arr.shape[:2]
    if n == 0:
        return 0, h, 0, w
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ys, xs = np.nonzero(big)
    return (max(ys.min() * 4 - 4, 0), min(ys.max() * 4 + 8, h),
            max(xs.min() * 4 - 4, 0), min(xs.max() * 4 + 8, w))


def normalise_image(src: bytes | Path, side: int = NORM_SIDE, q: int = NORM_JPEG_Q) -> tuple[bytes, dict]:
    """Crop to field of view, pad square, Lanczos to side x side, JPEG q95 4:4:4,
    no metadata.  Returns (jpeg_bytes, info) where info records native geometry."""
    import numpy as np
    from PIL import Image
    im = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    im.load()
    im = im.convert("RGB")
    arr = np.asarray(im)
    y0, y1, x0, x1 = fov_bbox(arr)
    crop = arr[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s, 3), dtype=crop.dtype)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    out = Image.fromarray(sq)
    if out.size != (side, side):
        out = out.resize((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=q, subsampling=0, optimize=False)
    info = {"native_w": int(arr.shape[1]), "native_h": int(arr.shape[0]),
            "fov_w": int(w), "fov_h": int(h), "fov_diameter": int(max(h, w))}
    return buf.getvalue(), info


# ---------------------------------------------------------------------------
# Dataset loaders: each returns a DataFrame with columns
#   source_file, grade (0..4, or 5 for ungradable), patient (or ""), quality, extra...
# plus a fetch(source_file) -> bytes function.  Loaders read labels only;
# they never send anything anywhere.
# ---------------------------------------------------------------------------
def _norm_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_aptos():
    import pandas as pd
    cfg = SOURCES["aptos"]
    df = _norm_cols(pd.read_csv(cfg["labels"], dtype=str))
    df = df.rename(columns={"id_code": "stem", "diagnosis": "grade"})
    df["grade"] = df.grade.astype(int)
    files = {p.stem: p for p in Path(cfg["images"]).glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    df = df[df.stem.isin(files)].copy()
    df["source_file"] = df.stem.map(lambda s: str(files[s]))
    df["patient"] = ""
    df["quality"] = ""
    return df[["source_file", "stem", "grade", "patient", "quality"]], lambda f: Path(f).read_bytes()


def paper1_aptos_stems() -> set[str]:
    """Stems of the 200 APTOS images used in paper 1, to exclude them here."""
    import pandas as pd
    p = SOURCES["aptos"]["paper1_key"]
    if not Path(p).exists():
        raise FileNotFoundError(f"paper 1 APTOS key not found at {p}; set SOURCES['aptos']['paper1_key']")
    k = pd.read_csv(p, dtype=str, keep_default_na=False)
    stems = set()
    for c in k.columns:
        vals = k[c].astype(str).map(lambda v: Path(v).stem)
        if vals.str.fullmatch(r"[0-9a-f]{12}").mean() > 0.9:   # APTOS id_codes are 12 hex chars
            stems |= set(vals)
    if len(stems) < 150:
        raise ValueError(f"could not find APTOS id_codes in {p}: columns {list(k.columns)}")
    return stems


class _ConcatFile(io.RawIOBase):
    """Read-only, seekable view of several byte ranges laid end to end (no copying)."""

    def __init__(self, segments):            # [(path, start, length), ...]
        self.segs, off = [], 0
        for path, start, length in segments:
            self.segs.append((open(path, "rb"), start, length, off))
            off += length
        self.size, self.pos = off, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b):
        n, done = len(b), 0
        while done < n and self.pos < self.size:
            for fh, start, length, off in self.segs:
                if off <= self.pos < off + length:
                    k = min(n - done, off + length - self.pos)
                    fh.seek(start + self.pos - off)
                    chunk = fh.read(k)
                    b[done:done + len(chunk)] = chunk
                    done += len(chunk)
                    self.pos += len(chunk)
                    break
            else:
                break
        return done


EYEPACS_MIN_IMAGES = 30000
EYEPACS_PART_DIR = Path("/content/eyepacs_parts")


def _eyepacs_zip(cfg):
    """Open the EyePACS training archive (Kaggle split zip, train.zip.001 to .005).
    On this Drive, parts .001 and .005 are raw, while .002 to .004 were saved inside
    a wrapper zip each (as the Kaggle downloader does).  Wrapped parts are unpacked once
    to local disk (about 25 GB); the five raw parts are then read end to end as one
    archive without further copying.  train_combined.zip (all five wrappers glued
    together) is not used."""
    import shutil
    import zipfile
    raw = Path(cfg["zip"]).parent
    EYEPACS_PART_DIR.mkdir(parents=True, exist_ok=True)
    segs = []
    for k in range(1, 6):
        name = f"train.zip.00{k}"
        f = raw / name
        if not f.exists():
            raise FileNotFoundError(f"missing EyePACS part {f}")
        with open(f, "rb") as fh:
            head = fh.read(64)
        wrapped = head[:4] == b"PK\x03\x04" and head[30:30 + len(name)] == name.encode()
        if wrapped:
            w = zipfile.ZipFile(f)
            info = w.getinfo(name)
            local = EYEPACS_PART_DIR / name
            if not local.exists() or local.stat().st_size != info.file_size:
                free = shutil.disk_usage(EYEPACS_PART_DIR).free
                assert free > info.file_size + (2 << 30), f"not enough local disk to unpack {name}"
                print(f"unpacking {name} ({info.file_size / 1e9:.1f} GB) to local disk ...")
                with w.open(info) as fi, open(local, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 24)
            f = local
        segs.append((f, 0, Path(f).stat().st_size))
    cf = _ConcatFile(segs)
    z = zipfile.ZipFile(io.BufferedReader(cf, buffer_size=1 << 20))
    n = sum(1 for x in z.namelist() if x.lower().endswith((".jpeg", ".jpg", ".png")))
    assert n > EYEPACS_MIN_IMAGES, f"rebuilt EyePACS archive has only {n} images"
    return z


def load_eyepacs():
    import pandas as pd
    cfg = SOURCES["eyepacs"]
    df = _norm_cols(pd.read_csv(cfg["labels"], dtype=str)).rename(columns={"image": "stem", "level": "grade"})
    df["grade"] = df.grade.astype(int)
    zf = _eyepacs_zip(cfg)
    members = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith((".jpeg", ".jpg", ".png"))}
    df = df[df.stem.isin(members)].copy()
    if df.empty:
        raise ValueError(f"no EyePACS label stems matched archive members, e.g. {list(members)[:5]}")
    df["source_file"] = df.stem.map(members)
    df["patient"] = df.stem.str.split("_").str[0]
    df["quality"] = ""
    return df[["source_file", "stem", "grade", "patient", "quality"]], lambda f: zf.read(f)


def load_idrid():
    """IDRiD disease-grading subset.  Training (413) and testing (103) images reuse the same
    file names (IDRiD_001 ...), so each label file is matched to its own image folder and the
    stem is prefixed with the split."""
    import pandas as pd
    root = Path(SOURCES["idrid"]["root"])
    parts = []
    for lab in root.rglob("*.csv"):
        name = lab.name.lower()
        if "disease grading" not in name or "label" not in name:
            continue
        split = "train" if "training" in name else "test"
        sub = "a. Training Set" if split == "train" else "b. Testing Set"
        img_dir = next(p for p in lab.parents if p.name.endswith("Disease Grading") and (p / "1. Original Images").exists()) \
            / "1. Original Images" / sub
        d = _norm_cols(pd.read_csv(lab, dtype=str)).iloc[:, :2].set_axis(["name", "grade"], axis=1).dropna()
        d["name"] = d.name.str.strip()
        files = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")}
        d = d[d.name.isin(files)].copy()
        d["source_file"] = d.name.map(lambda n: str(files[n]))
        d["stem"] = split + "_" + d.name
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"no IDRiD disease grading label csv under {root}")
    df = pd.concat(parts, ignore_index=True)
    df["grade"] = df.grade.astype(int)
    df["patient"] = ""
    df["quality"] = ""
    return df[["source_file", "stem", "grade", "patient", "quality"]], lambda f: Path(f).read_bytes()


def load_ddr():
    import pandas as pd
    root = Path(SOURCES["ddr"]["root"])
    rows = []
    for txt in root.rglob("*.txt"):
        if txt.stem not in ("train", "valid", "test") or "grading" not in str(txt.parent).lower():
            continue
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                rows.append({"name": parts[0], "grade": int(parts[1]), "split": txt.stem, "dir": txt.parent / txt.stem})
    if not rows:
        raise FileNotFoundError(f"no DDR grading txt files under {root}")
    df = pd.DataFrame(rows)
    df["source_file"] = [str(d / n) for d, n in zip(df["dir"], df["name"])]
    df = df[[Path(f).exists() for f in df.source_file]].copy()
    df["stem"] = df.name.map(lambda n: Path(n).stem)
    df["patient"] = ""
    df["quality"] = df.grade.map(lambda g: "ungradable" if g == 5 else "")
    return df[["source_file", "stem", "grade", "patient", "quality"]], lambda f: Path(f).read_bytes()


def load_sustech():
    import pandas as pd
    import zipfile
    cfg = SOURCES["sustech"]
    raw = Path(cfg["local"])
    if not any(raw.rglob("drLabels.csv")):
        raw.mkdir(parents=True, exist_ok=True)
        zp = raw / "sustech.zip"
        if not zp.exists():
            import gdown  # type: ignore
            gdown.download(id=cfg["gdrive_id"], output=str(zp), quiet=False)
        with zipfile.ZipFile(zp) as z:
            z.extractall(raw)
        for inner in list(raw.rglob("*.zip")):
            if inner != zp:
                with zipfile.ZipFile(inner) as z:
                    z.extractall(inner.parent)
    lbl = next(raw.rglob("drLabels.csv"))
    lab = pd.read_csv(lbl).iloc[:, :3].set_axis(["file", "eye", "grade"], axis=1)
    lab["stem"] = lab.file.astype(str).str.strip().map(lambda f: Path(f).stem)
    lab["grade"] = pd.to_numeric(lab.grade, errors="coerce")
    lab = lab.dropna(subset=["grade"])
    lab["grade"] = lab.grade.astype(int)
    lab = lab[lab.grade <= 4].copy()          # grade 5 = laser-treated eyes, excluded
    files = {p.stem: p for p in lbl.parent.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")}
    lab = lab[lab.stem.isin(files)].copy()
    lab["source_file"] = lab.stem.map(lambda s: str(files[s]))
    lab["patient"] = ""
    lab["quality"] = ""
    return lab[["source_file", "stem", "grade", "patient", "quality"]], lambda f: Path(f).read_bytes()


def load_brset():
    """BRSET: labels from label_brset.csv; images streamed from the PhysioNet tar.
    Runs only on our own compute.  Nothing here is sent anywhere."""
    import pandas as pd
    import tarfile
    cfg = SOURCES["brset"]
    df = _norm_cols(pd.read_csv(cfg["labels"], dtype=str, keep_default_na=False))
    need = {"image_id", "patient_id", "DR_ICDR", "quality"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"BRSET label columns missing {missing}; have {list(df.columns)[:30]}")
    df["grade"] = pd.to_numeric(df.DR_ICDR, errors="coerce")
    df = df.dropna(subset=["grade"]).copy()
    df["grade"] = df.grade.astype(int)
    df["stem"] = df.image_id.astype(str)
    df["patient"] = df.patient_id.astype(str)
    df["quality"] = df.quality.astype(str).str.strip().str.lower()
    df.loc[df.quality == "inadequate", "grade"] = 5   # ungradable pool
    tf = tarfile.open(cfg["tar"])
    members = {Path(m.name).stem: m for m in tf.getmembers() if m.isfile()
               and m.name.lower().endswith((".jpg", ".jpeg", ".png"))}
    df = df[df.stem.isin(members)].copy()
    df["source_file"] = df.stem
    return (df[["source_file", "stem", "grade", "patient", "quality"]],
            lambda f: tf.extractfile(members[f]).read())


LOADERS = {"aptos": load_aptos, "eyepacs": load_eyepacs, "idrid": load_idrid,
           "ddr": load_ddr, "sustech": load_sustech, "brset": load_brset}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def stratified_sample(df, dataset: str, seed: int = SEED, n_per_grade: int = N_PER_GRADE,
                      n_ungradable: int = 0, exclude_stems: set | None = None):
    """Up to n_per_grade images per grade 0..4 (one per patient where known) plus
    n_ungradable images of grade 5.  Deterministic given seed and the label table."""
    import pandas as pd
    d = df.copy()
    if exclude_stems:
        d = d[~d.stem.isin(exclude_stems)]
    d = d.sort_values("stem").reset_index(drop=True)            # order-independent of file listing
    rng = random.Random(f"{seed}|{dataset}")
    picks = []
    used_patients: set[str] = set()
    for g in [0, 1, 2, 3, 4, 5]:
        n = n_ungradable if g == 5 else n_per_grade
        if n <= 0:
            continue
        pool = d[d.grade == g].index.tolist()
        rng.shuffle(pool)
        chosen = []
        for i in pool:
            p = d.at[i, "patient"]
            if p and p in used_patients:
                continue
            chosen.append(i)
            if p:
                used_patients.add(p)
            if len(chosen) == n:
                break
        picks.extend(chosen)
    out = d.loc[picks].copy()
    out["dataset"] = dataset
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task queue
# ---------------------------------------------------------------------------
def task_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:20]


def build_stage1_queue(registry, seed: int = SEED, vendors: list[str] | None = None) -> list[dict]:
    """registry: DataFrame with image_id, dataset.  No ground truth by construction.
    Restricted datasets go to local models only."""
    vendors = vendors or ALL_IMAGE_MODELS
    rows = []
    for r in registry.itertuples():
        for v in vendors:
            if r.dataset in RESTRICTED and v not in LOCAL_VENDORS:
                continue
            for pr, k in K_REPEATS.items():
                for rep in range(k):
                    rows.append({"task_id": task_id("s1", v, r.image_id, pr, rep), "stage": 1, "arm": pr,
                                 "vendor": v, "image_id": r.image_id, "dataset": r.dataset, "prompt_id": pr,
                                 "rep": rep, "prompt": PROMPTS[pr], "with_image": True})
    ids = [t["task_id"] for t in rows]
    assert len(ids) == len(set(ids)), "task_id collision"
    random.Random(seed).shuffle(rows)
    for i, t in enumerate(rows):
        t["queue_pos"] = i
    return rows


BRSET_THIRD_PARTY_OK = False   # set True only after written PhysioNet approval is logged in config


def build_stage2_queue(stage1_records: list[dict], seed: int = SEED, brset_third_party_ok: bool | None = None) -> list[dict]:
    """From the frozen stage-1 free-text records (rep 0, ok, non-empty text) build:
      jev_text  : Jev reads the description (third party; restricted datasets only if approved)
      self_text : the same model reads its own description, text only (local for qwen)
    The rule-based mapper needs no calls.  Contains no ground truth."""
    ok3p = BRSET_THIRD_PARTY_OK if brset_third_party_ok is None else brset_third_party_ok
    rows = []
    for r in stage1_records:
        if r.get("prompt_id") != "free" or not r.get("ok") or not (r.get("text") or "").strip():
            continue
        text = strip_think(r["text"])
        base = {"stage": 2, "image_id": r["image_id"], "dataset": r["dataset"], "source_vendor": r["vendor"],
                "source_task_id": r["task_id"], "rep": r.get("rep", 0), "with_image": False}
        if r["dataset"] not in RESTRICTED or ok3p:
            rows.append({**base, "task_id": task_id("s2", "jev_text", r["task_id"]), "arm": "jev_text",
                         "vendor": "jev", "prompt_id": "jev_text", "prompt": text})
        rows.append({**base, "task_id": task_id("s2", "self_text", r["task_id"]), "arm": "self_text",
                     "vendor": r["vendor"], "prompt_id": "self_text", "prompt": SELF_PASS_PROMPT + text})
    random.Random(seed + 2).shuffle(rows)
    for i, t in enumerate(rows):
        t["queue_pos"] = i
    return rows


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(text: str | None) -> str | None:
    return None if text is None else _THINK.sub("", text).strip()


def parse_json_reply(text: str | None) -> tuple[dict | None, str]:
    if not text:
        return None, "empty"
    t = _FENCE.sub("", strip_think(text)).strip()
    try:
        obj = json.loads(t)
        return (obj, "ok") if isinstance(obj, dict) else (None, "not_object")
    except Exception:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        if isinstance(obj, dict):
                            return obj, "recovered"
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None, "unparseable"


def norm_grade(v) -> str | None:
    """Map a model's grade field to one of GRADES, or None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() and 0 <= v <= 4 else None
    s = str(v).strip().lower()
    if re.search(r"ungrad|cannot|not grad|insufficient|poor quality", s):
        return "ungradable"
    m = re.fullmatch(r"(?:grade\s*)?([0-4])(?:\.0)?", s)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Rule-based mapper: findings text -> ICDR grade (fixed before any data is seen)
# ---------------------------------------------------------------------------
_NEG = r"(?:no|without|absence of|absent|not seen|none|negative for|free of|no evidence of|no signs? of|nor)"
_LESION = {
    "nv": r"neovasc\w*|new vessels?|\bnv[de]\b|fibrovascular|proliferative (?:diabetic )?retinopathy|\bpdr\b",
    "vh": r"vitreous ha?emorrhage|pre-?retinal ha?emorrhage|subhyaloid ha?emorrhage",
    "beading": r"venous beading|beaded veins?",
    "irma": r"intraretinal microvascular|\birma\b",
    "heme": r"ha?emorrhages?|blot|flame|dot[- ]and[- ]blot",
    "ma": r"micro-?aneurysms?",
    "exudate": r"(?:hard )?exudates?|lipid",
    "cws": r"cotton[- ]wool",
    "laser": r"laser (?:scars?|spots?|marks?)|panretinal photocoag\w*|\bprp\b|photocoagulation scars?",
    "ungradable": r"ungradable|cannot be (?:graded|assessed)|too (?:blurry|dark|poor)|"
                  r"poor (?:image )?quality|non-?diagnostic|inadequate (?:image|quality)",
    "normal": r"no (?:signs? of |evidence of |apparent )?diabetic retinopathy|normal fundus|"
              r"unremarkable|within normal limits",
}
_FOUR_Q = r"(?:all )?(?:four|4) quadrants|every quadrant|all quadrants|extensive|numerous|widespread|severe"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+|\n+", text) if s.strip()]


def _present(text: str, pat: str) -> bool:
    """True if pattern appears in some sentence without a preceding negation in that clause."""
    for s in _sentences(text.lower()):
        for m in re.finditer(pat, s):
            # negation scope runs to the end of the clause; commas stay in scope ("no MAs, haemorrhages or
            # exudates"), contrastive words and positive presentatives end it
            clause = re.split(r";|\bbut\b|\bhowever\b|\balthough\b|\bexcept\b|\bwhile\b|\bthere (?:are|is)\b"
                              r"|\bshows?\b|\bpresent\b|\bseen\b(?! in)", s[:m.start()])[-1]
            if not re.search(rf"\b{_NEG}\b", clause):
                return True
    return False


def rule_grade(text: str | None) -> tuple[str | None, dict]:
    """Deterministic ICDR grade from a free-text findings description."""
    if not text:
        return None, {}
    t = strip_think(text)
    f = {k: _present(t, p) for k, p in _LESION.items()}
    low = t.lower()
    four_q = bool(re.search(_FOUR_Q, low))
    if f["ungradable"] and not any(f[k] for k in ("nv", "vh", "heme", "ma", "exudate", "beading", "irma")):
        g = "ungradable"
    elif f["nv"] or f["vh"]:
        g = "4"
    elif f["beading"] or f["irma"] or (f["heme"] and four_q):
        g = "3"
    elif f["heme"] or f["exudate"] or f["cws"] or f["laser"]:
        g = "2"
    elif f["ma"]:
        g = "1"
    elif f["normal"] or not any(f.values()):
        g = "0"
    else:
        g = "0"
    f["four_quadrants"] = four_q
    return g, f


# ---------------------------------------------------------------------------
# Vendor adapters (raw REST; image optional so the same code does text-only calls)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CallResult:
    http_status: int | None
    ok: bool
    text: str | None
    served_model: str | None
    finish_reason: str | None
    usage: dict | None
    raw: dict | None
    error: str | None
    retryable: bool


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def build_request(vendor: str, model: str, prompt: str, jpeg: bytes | None, api_key: str):
    url, headers, body = _build_request_base(vendor, model, prompt, jpeg, api_key)
    extra = EXTRA_BODY.get(vendor) or {}
    if extra:
        body = json.loads(json.dumps(body))
        if vendor == "google":
            body["generationConfig"].update(extra)
        else:
            body.update(extra)
    return url, headers, body


def _build_request_base(vendor, model, prompt, jpeg, api_key):
    if vendor == "anthropic":
        content = ([{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(jpeg)}}]
                   if jpeg else []) + [{"type": "text", "text": prompt}]
        return ("https://api.anthropic.com/v1/messages",
                {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                {"model": model, "max_tokens": MAX_OUTPUT_TOKENS, "messages": [{"role": "user", "content": content}]})
    if vendor == "openai":
        content = ([{"type": "input_image", "image_url": "data:image/jpeg;base64," + _b64(jpeg), "detail": "high"}]
                   if jpeg else []) + [{"type": "input_text", "text": prompt}]
        return ("https://api.openai.com/v1/responses",
                {"authorization": f"Bearer {api_key}", "content-type": "application/json"},
                {"model": model, "max_output_tokens": MAX_OUTPUT_TOKENS, "store": False,
                 "input": [{"role": "user", "content": content}]})
    if vendor == "google":
        parts = ([{"inline_data": {"mime_type": "image/jpeg", "data": _b64(jpeg)}}] if jpeg else []) + [{"text": prompt}]
        return (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                {"x-goog-api-key": api_key, "content-type": "application/json"},
                {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS}})
    if vendor == "xai":
        content = ([{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(jpeg), "detail": "high"}}]
                   if jpeg else []) + [{"type": "text", "text": prompt}]
        return ("https://api.x.ai/v1/chat/completions",
                {"authorization": f"Bearer {api_key}", "content-type": "application/json"},
                {"model": model, "max_tokens": MAX_OUTPUT_TOKENS, "messages": [{"role": "user", "content": content}]})
    raise ValueError(vendor)


def extract(vendor: str, j: dict):
    if vendor == "anthropic":
        text = "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text") or None
        return text, j.get("model"), j.get("stop_reason"), j.get("usage")
    if vendor == "openai":
        parts = [c.get("text", "") for it in (j.get("output") or []) if it.get("type") == "message"
                 for c in (it.get("content") or []) if c.get("type") == "output_text"]
        text = "".join(parts) or j.get("output_text") or None
        fin = j.get("status")
        if j.get("incomplete_details"):
            fin = f"{fin}:{j['incomplete_details'].get('reason')}"
        return text, j.get("model"), fin, j.get("usage")
    if vendor == "google":
        cands = j.get("candidates") or []
        text, fin = None, None
        if cands:
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
            fin = cands[0].get("finishReason")
        elif j.get("promptFeedback"):
            fin = "blocked:" + str(j["promptFeedback"].get("blockReason"))
        return text, j.get("modelVersion"), fin, j.get("usageMetadata")
    if vendor == "xai":
        ch = (j.get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content"), j.get("model"), ch.get("finish_reason"), j.get("usage")
    raise ValueError(vendor)


def _truncate_raw(j: dict, limit: int = 60000) -> dict:
    s = json.dumps(j)
    return j if len(s) <= limit else {"_truncated": True, "_len": len(s), "head": s[:limit]}


async def _post(client, url, headers, body) -> tuple[int | None, dict | None, str | None]:
    try:
        r = await client.post(url, headers=headers, json=body, timeout=HTTP_TIMEOUT_S)
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
    try:
        return r.status_code, r.json(), None
    except Exception:
        return r.status_code, {"_non_json": r.text[:2000]}, None


_RETRY = (408, 409, 425, 429, 500, 502, 503, 504, 529)


async def call_once(client, vendor, model, prompt, jpeg, api_key) -> CallResult:
    url, headers, body = build_request(vendor, model, prompt, jpeg, api_key)
    status, j, err = await _post(client, url, headers, body)
    if err:
        return CallResult(None, False, None, None, None, None, None, err, True)
    if status != 200:
        return CallResult(status, False, None, None, None, None, _truncate_raw(j), f"HTTP {status}", status in _RETRY)
    text, served, fin, usage = extract(vendor, j)
    return CallResult(status, True, text, served, fin, usage, _truncate_raw(j), None, False)


# ---- Jev (OpenRouter Decisions API) ----------------------------------------
def jev_body(state, model: str = JEV_MODEL) -> dict:
    return {"model": model, "state": state,
            "questions": {"icdr": {"type": "choice", "instructions": JEV_INSTRUCTIONS,
                                   "criteria": dict(ICDR_TEXT)}}}


def find_choice_answer(j: Any) -> dict | None:
    """Locate the Choice answer (has 'probabilities' and 'choice') anywhere in the response."""
    if isinstance(j, dict):
        if "probabilities" in j and ("choice" in j or j.get("type") == "choice"):
            return j
        for v in j.values():
            r = find_choice_answer(v)
            if r:
                return r
    elif isinstance(j, list):
        for v in j:
            r = find_choice_answer(v)
            if r:
                return r
    return None


async def call_jev(client, state, api_key, model: str = JEV_MODEL) -> CallResult:
    headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
    status, j, err = await _post(client, JEV_URL, headers, jev_body(state, model))
    if err:
        return CallResult(None, False, None, None, None, None, None, err, True)
    if status != 200:
        return CallResult(status, False, None, None, None, None, _truncate_raw(j), f"HTTP {status}", status in _RETRY)
    ans = find_choice_answer(j)
    served = j.get("model") if isinstance(j, dict) else None
    return CallResult(status, ans is not None, json.dumps(ans) if ans else None, served,
                      None if ans else "no_choice_answer", (j or {}).get("usage"), _truncate_raw(j),
                      None if ans else "no choice answer in response", False)


# ---------------------------------------------------------------------------
# Runner (interleaved, resumable, served-model lock), API vendors + Jev
# ---------------------------------------------------------------------------
def load_done(calls_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if Path(calls_path).exists():
        with open(calls_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done[rec["task_id"]] = rec
                    except Exception:
                        pass
    return done


class Runner:
    """Each task supplies: task_id, vendor ('anthropic'|'openai'|'google'|'xai'|'jev'),
    prompt (str), and either image_id (sent with the image) or no image.
    Guarantees: an ok task is never re-sent; failures are retried on the next run;
    the first served-model string per vendor is locked, and a change pauses that vendor."""

    def __init__(self, run_dir: Path, models: dict[str, str], image_dir: Path | None = None,
                 concurrency: dict[str, int] | None = None):
        self.run_dir = Path(run_dir)
        self.calls_path = self.run_dir / "calls.jsonl"
        self.models = models
        self.image_dir = Path(image_dir) if image_dir else None
        self.conc = {**DEFAULT_CONCURRENCY, **(concurrency or {})}
        self.locked = read_json(self.run_dir / "served_model_lock.json") if (self.run_dir / "served_model_lock.json").exists() else {}
        self.paused: dict[str, str] = {}
        self._fh = None
        self.keys: dict[str, str] = {}
        self.counts: dict[str, dict] = {}

    def _key(self, vendor):
        kv = "openrouter" if vendor == "jev" else vendor
        if kv not in self.keys:
            k = get_api_key(kv)
            if not k:
                raise RuntimeError(f"missing API key for {kv}")
            self.keys[kv] = k
        return self.keys[kv]

    def _check_lock(self, vendor, served) -> bool:
        if not served:
            return True
        if vendor not in self.locked:
            self.locked[vendor] = served
            write_json(self.run_dir / "served_model_lock.json", self.locked)
            return True
        if served != self.locked[vendor]:
            self.paused[vendor] = f"served model changed: {self.locked[vendor]} -> {served}"
            return False
        return True

    async def _one(self, client, sem, task):
        v = task["vendor"]
        async with sem[v]:
            if v in self.paused:
                return
            jpeg = (self.image_dir / f"{task['image_id']}.jpg").read_bytes() if task.get("with_image") else None
            t0 = time.time()
            attempt = 0
            while True:
                attempt += 1
                if v == "jev":
                    res = await call_jev(client, task["prompt"] if not jpeg else
                                         [{"type": "image", "data": _b64(jpeg), "media_type": "image/jpeg"},
                                          task["prompt"]], self._key(v), self.models["jev"])
                else:
                    res = await call_once(client, v, self.models[v], task["prompt"], jpeg, self._key(v))
                if res.ok or not res.retryable or attempt >= MAX_ATTEMPTS:
                    break
                await asyncio.sleep(min(90, 2 ** attempt + random.random() * 2))
            parsed, pstat = parse_json_reply(res.text) if res.ok else (None, "no_response")
            lock_ok = self._check_lock(v, res.served_model) if res.ok else True
            rec = {k: task.get(k) for k in ("task_id", "stage", "arm", "vendor", "image_id", "dataset", "prompt_id",
                                             "rep", "source_task_id", "queue_pos")}
            rec.update({"model_requested": self.models[v], "served_model": res.served_model,
                        "served_model_matches_lock": lock_ok, "prompt_sha": sha256_bytes(task["prompt"].encode())[:16],
                        "image_sha": sha256_bytes(jpeg)[:16] if jpeg else None, "t_start": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
                        "latency_s": round(time.time() - t0, 2), "attempts": attempt, "http_status": res.http_status,
                        "ok": bool(res.ok and lock_ok), "finish_reason": res.finish_reason, "text": res.text,
                        "parsed": parsed, "parse_status": pstat, "usage": res.usage,
                        "error": res.error if lock_ok else self.paused.get(v), "raw": res.raw})
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()
            c = self.counts.setdefault(v, {"ok": 0, "fail": 0})
            c["ok" if rec["ok"] else "fail"] += 1

    async def run(self, queue: list[dict], progress_every: int = 200, limit: int | None = None) -> dict:
        import httpx
        done = load_done(self.calls_path)
        todo = [t for t in queue if not done.get(t["task_id"], {}).get("ok")]
        if limit:
            todo = todo[:limit]
        print(f"{len(done)} tasks already logged; {len(todo)} to send")
        vendors = sorted({t["vendor"] for t in todo})
        sem = {v: asyncio.Semaphore(self.conc.get("openrouter" if v == "jev" else v, 4)) for v in vendors}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.calls_path, "a")
        try:
            async with httpx.AsyncClient(limits=httpx.Limits(max_connections=sum(self.conc.values()) + 4)) as client:
                pending, n = set(), 0
                for t in todo:
                    pending.add(asyncio.create_task(self._one(client, sem, t)))
                    if len(pending) >= 4 * sum(self.conc.values()):
                        fin, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        n += len(fin)
                        if n % progress_every < len(fin):
                            print(f"[{utc_now()[:19]}] {n}/{len(todo)} {self.counts} paused={self.paused}")
                if pending:
                    await asyncio.wait(pending)
        finally:
            self._fh.close()
        print(f"finished. {self.counts} paused={self.paused}")
        return {"counts": self.counts, "paused": self.paused, "locked": self.locked}


def summarise(calls_path: Path):
    import pandas as pd
    done = load_done(Path(calls_path))
    if not done:
        print("no calls yet")
        return None
    df = pd.DataFrame([{k: r.get(k) for k in ("vendor", "prompt_id", "arm", "ok", "parse_status", "served_model",
                                               "latency_s", "dataset")} for r in done.values()])
    tab = df.groupby(["vendor"] + [c for c in ("prompt_id", "arm") if df[c].notna().any()]).agg(
        n=("ok", "size"), ok=("ok", "mean"),
        parsed=("parse_status", lambda s: s.isin(["ok", "recovered"]).mean()),
        p50_latency=("latency_s", "median")).round(3)
    rate = df.groupby("vendor").ok.mean().round(4)
    return {"table": tab, "success_rate": rate, "below_floor": list(rate[rate < SUCCESS_RATE_FLOOR].index),
            "served_models": df.groupby("vendor").served_model.agg(lambda s: sorted({x for x in s if isinstance(x, str)}))}


# ---------------------------------------------------------------------------
# Local open-weight model (vLLM).  Runs on our own GPU; nothing leaves the machine.
# ---------------------------------------------------------------------------
def qwen_messages(prompt: str, jpeg: bytes | None) -> list[dict]:
    content = ([{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(jpeg)}}] if jpeg else [])
    return [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]


def run_local(run_dir: Path, queue: list[dict], image_dir: Path, model_id: str, revision: str | None = None,
              batch: int = 32, max_tokens: int = MAX_OUTPUT_TOKENS, image_dirs: dict | None = None) -> dict:
    """Resumable batched inference with vLLM.  Uses the model's own generation
    defaults (the open-weight analogue of 'vendor defaults')."""
    from vllm import LLM  # type: ignore
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    calls = run_dir / "calls.jsonl"
    done = load_done(calls)
    todo = [t for t in queue if t["vendor"] == "qwen" and not done.get(t["task_id"], {}).get("ok")]
    print(f"{len(done)} logged; {len(todo)} to run")
    if not todo:
        return {"todo": 0}
    llm = LLM(model=model_id, revision=revision, max_model_len=32768, limit_mm_per_prompt={"image": 1},
              trust_remote_code=False)
    sp = llm.get_default_sampling_params()
    sp.max_tokens = max_tokens
    with open(calls, "a") as fh:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            msgs, jpegs = [], []
            for t in chunk:
                d = (image_dirs or {}).get(t["dataset"], image_dir)
                jb = (Path(d) / f"{t['image_id']}.jpg").read_bytes() if t.get("with_image") else None
                jpegs.append(jb)
                msgs.append(qwen_messages(t["prompt"], jb))
            t0 = time.time()
            outs = llm.chat(msgs, sp, use_tqdm=False)
            dt = (time.time() - t0) / max(len(chunk), 1)
            for t, o, jb in zip(chunk, outs, jpegs):
                text = o.outputs[0].text
                parsed, pstat = parse_json_reply(text) if t.get("prompt_id") != "free" else (None, "text")
                rec = {k: t.get(k) for k in ("task_id", "stage", "arm", "vendor", "image_id", "dataset", "prompt_id",
                                             "rep", "source_task_id", "queue_pos")}
                rec.update({"model_requested": model_id, "served_model": f"{model_id}@{revision or 'main'}",
                            "served_model_matches_lock": True, "prompt_sha": sha256_bytes(t["prompt"].encode())[:16],
                            "image_sha": sha256_bytes(jb)[:16] if jb else None, "t_start": utc_now(), "latency_s": round(dt, 2),
                            "attempts": 1, "http_status": None, "ok": True,
                            "finish_reason": o.outputs[0].finish_reason, "text": text, "parsed": parsed,
                            "parse_status": pstat, "error": None,
                            "usage": {"prompt_tokens": len(o.prompt_token_ids or []),
                                      "completion_tokens": len(o.outputs[0].token_ids or [])}})
                fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(f"[{utc_now()[:19]}] {min(i + batch, len(todo))}/{len(todo)}")
    return {"todo": len(todo)}


# ---------------------------------------------------------------------------
# Exploratory: Jev reads the image directly (only if the probe shows it works)
# ---------------------------------------------------------------------------
JEV_IMAGE_STATE_TEXT = "Retinal fundus photograph from a diabetic retinopathy screening programme."


def build_jev_image_queue(registry, seed: int = SEED) -> list[dict]:
    rows = []
    for r in registry.itertuples():
        if r.dataset in RESTRICTED:
            continue
        rows.append({"task_id": task_id("s2", "jev_image", r.image_id), "stage": 2, "arm": "jev_image",
                     "vendor": "jev", "image_id": r.image_id, "dataset": r.dataset, "prompt_id": "jev_image",
                     "rep": 0, "prompt": JEV_IMAGE_STATE_TEXT, "with_image": True})
    random.Random(seed + 3).shuffle(rows)
    for i, t in enumerate(rows):
        t["queue_pos"] = i
    return rows


# ---------------------------------------------------------------------------
# GitHub: commit-once (a path that exists with different content is an error,
# never an overwrite).  Token is the Colab secret GITHUB_TOKEN.
# ---------------------------------------------------------------------------
GH_REPO = "avnish-deobhakta/motion-canonical"
GH_PREFIX = "dr6/"


def _gh():
    import httpx
    from google.colab import userdata  # type: ignore
    tok = userdata.get("GITHUB_TOKEN")
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    r = httpx.get(f"https://api.github.com/repos/{GH_REPO}", headers=h, timeout=30)
    assert r.status_code == 200, f"GitHub HTTP {r.status_code}: give this notebook GITHUB_TOKEN access, or renew the token"
    return httpx, h, r.json()["default_branch"]


def gh_get(path: str) -> bytes | None:
    httpx, h, br = _gh()
    g = httpx.get(f"https://api.github.com/repos/{GH_REPO}/contents/{GH_PREFIX}{path}", headers=h,
                  params={"ref": br}, timeout=30)
    if g.status_code == 404:
        return None
    g.raise_for_status()
    return base64.b64decode(g.json()["content"])


def gh_commit_once(path: str, data: bytes | str, message: str) -> str:
    """Create GH_PREFIX+path.  Same content already there: no-op.  Different: raise."""
    if isinstance(data, str):
        data = data.encode()
    httpx, h, br = _gh()
    existing = gh_get(path)
    if existing is not None:
        if existing == data:
            print(f"already committed, identical: {GH_PREFIX}{path}")
            return "unchanged"
        raise RuntimeError(f"{GH_PREFIX}{path} already exists with different content; write an amendment instead")
    r = httpx.put(f"https://api.github.com/repos/{GH_REPO}/contents/{GH_PREFIX}{path}", headers=h, timeout=60,
                  json={"message": message, "content": base64.b64encode(data).decode(), "branch": br})
    r.raise_for_status()
    sha = r.json()["commit"]["sha"]
    print(f"committed {GH_PREFIX}{path} @ {sha[:10]}")
    return sha


def freeze_file(p: Path) -> dict:
    p = Path(p)
    n = sum(1 for _ in open(p)) if p.exists() else 0
    return {"path": str(p).replace(str(MYDRIVE) + "/", ""), "sha256": sha256_file(p), "lines": n, "frozen_utc": utc_now()}
