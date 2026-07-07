"""
generate_drr_all.py

Generates 10 DRRs per CT for all datasets in Lumbar_ALL_LPS.
Output goes INSIDE each patient folder alongside ct.nii.gz and seg.nii.gz.

Structure after running:
  Lumbar_ALL_LPS/
    lumbar_0001/
      ct.nii.gz
      seg.nii.gz
      AP_0/
        drr_AP_0.png
        drr_AP_0.dcm
        P_AP_0.txt
        geometry_AP_0.jsons
      LP_90/
        drr_LP_90.png
        drr_LP_90.dcm
        P_LP_90.txt
        geometry_LP_90.json
      ... (10 view folders)
    lumbar_0002/
      ...

Resume-safe: skips CTs where all 10 views already have geometry JSON.
"""

import os, json, glob, subprocess, sys, time, argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image
import cv2
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import datetime

# ── CONFIGURATION ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate DRRs for all patients")
parser.add_argument("--input_dir", type=str, required=True,
                    help="Root directory containing patient folders (e.g., lumbar_0001/ct.nii.gz)")
parser.add_argument("--plastimatch", type=str, default="plastimatch",
                    help="Path to Plastimatch executable")
args = parser.parse_args()

ROOT            = args.input_dir
PLASTIMATCH_EXE = args.plastimatch

SAD = 1000.0
SID = 1500.0
IMAGE_DIM        = (1024, 1024)
DETECTOR_SIZE_MM = (500.0, 500.0)
BONE_MULTIPLIER  = 2.5

VIEW_DEFINITIONS = [
    ("OBL_35", 35), ("OBL_125", 125),
    ("OBL_55", 55), ("OBL_145", 145),
]


# ── HELPERS ────────────────────────────────────────────────────────────
def nrm_from_angle(angle_deg):
    rad = np.radians(float(angle_deg))
    return np.array([-np.sin(rad), np.cos(rad), 0.0])


def prepare_ct(ct_path, out_path):
    ct  = sitk.ReadImage(str(ct_path))
    arr = sitk.GetArrayFromImage(ct).astype(np.float32)
    arr[arr <= -800] = -1000
    bone = arr > 300
    arr[bone] = ((1.0 + arr[bone] / 1000.0) * BONE_MULTIPLIER - 1.0) * 1000.0
    out = sitk.GetImageFromArray(arr.astype(np.int16))
    out.SetSpacing(ct.GetSpacing())
    out.SetOrigin(ct.GetOrigin())
    out.SetDirection(ct.GetDirection())
    sitk.WriteImage(out, str(out_path))


def read_pfm(path):
    with open(path, 'rb') as f:
        hdr = f.readline().decode().strip()
        ch = 1 if hdr == 'Pf' else 3
        w, h = map(int, f.readline().decode().strip().split())
        scale = float(f.readline().decode().strip())
        data = np.frombuffer(f.read(), ('<' if scale < 0 else '>') + 'f')
        data = data.reshape((h, w) if ch == 1 else (h, w, ch))
    return np.flipud(data).astype(np.float32)


def save_png(arr, path):
    arr = arr.astype(np.float64)
    if arr.max() <= 0:
        Image.fromarray(np.zeros(IMAGE_DIM[::-1], dtype=np.uint8)).save(path)
        return
    xray = 1.0 - np.exp(-arr)
    valid = xray[xray > 0.001]
    p0, p1 = (np.percentile(valid, [0.5, 99.5]) if len(valid) > 100
              else (xray.min(), xray.max()))
    norm = np.clip((xray - p0) / max(p1 - p0, 1e-9), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    enh = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u8)
    Image.fromarray(enh, 'L').save(path)


def save_dcm(png_path, dcm_path, ct_id, view_label):
    """Save DRR as DICOM (grayscale, 1024x1024)."""
    img = np.array(Image.open(png_path).convert("L"))
    assert img.shape == (IMAGE_DIM[1], IMAGE_DIM[0]), f"Bad shape: {img.shape}"

    ds = Dataset()
    ds.file_meta = pydicom.dataset.FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"  # CR
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # Explicit VR Little Endian

    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientName = ct_id
    ds.PatientID = ct_id
    ds.Modality = "DX"
    ds.SeriesDescription = f"DRR {view_label}"
    ds.Rows, ds.Columns = img.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = img.tobytes()

    ds.save_as(str(dcm_path), write_like_original=False)


def parse_plastimatch_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    cx, cy = float(lines[0].split()[0]), float(lines[0].split()[1])
    P_norm = np.array([
        [float(v) for v in lines[1].split()],
        [float(v) for v in lines[2].split()],
        [float(v) for v in lines[3].split()],
    ])
    P_pix = P_norm.copy()
    P_pix[0] = P_norm[0] + cx * P_norm[2]
    P_pix[1] = P_norm[1] + cy * P_norm[2]
    return P_pix


def is_ct_done(ct_dir):
    """Check if all 10 views already have geometry JSON."""
    for label, _ in VIEW_DEFINITIONS:
        geo = os.path.join(ct_dir, label, f"geometry_{label}.json")
        if not os.path.exists(geo):
            return False
    return True


# ── MAIN ───────────────────────────────────────────────────────────────
def process_one_ct(ct_dir):
    ct_id   = os.path.basename(ct_dir)
    ct_path = os.path.join(ct_dir, "ct.nii.gz")

    if not os.path.exists(ct_path):
        return "no_ct"

    if is_ct_done(ct_dir):
        return "skip"

    # Verify LPS
    ct = sitk.ReadImage(ct_path)
    dirn = np.array(ct.GetDirection()).reshape(3, 3)
    if not np.allclose(dirn, np.eye(3), atol=1e-3):
        return "not_lps"

    size = ct.GetSize()
    center = [(s - 1) / 2.0 for s in size]
    iso = np.array(ct.TransformContinuousIndexToPhysicalPoint(center))

    # Bone-enhanced temp CT
    temp_ct = os.path.join(ct_dir, "_temp_ct.nii.gz")
    prepare_ct(ct_path, temp_ct)

    ok = 0
    for label, angle_deg in VIEW_DEFINITIONS:
        nrm = nrm_from_angle(angle_deg)
        source = iso - SAD * nrm

        view_dir = os.path.join(ct_dir, label)
        os.makedirs(view_dir, exist_ok=True)

        # Check if this view already done
        if os.path.exists(os.path.join(view_dir, f"geometry_{label}.json")):
            ok += 1
            continue

        prefix = os.path.join(view_dir, f"drr_{label}_")

        cmd = [
            PLASTIMATCH_EXE, "drr",
            "--algorithm", "exact",
            "--input", temp_ct,
            "--output", prefix,
            "--output-format", "pfm",
            "--sad", f"{SAD}",
            "--sid", f"{SID}",
            "--nrm", f"{nrm[0]:.6f} {nrm[1]:.6f} {nrm[2]:.6f}",
            "--vup", "0.0 0.0 1.0",
            "--isocenter", f"{iso[0]:.4f} {iso[1]:.4f} {iso[2]:.4f}",
            "--dim", f"{IMAGE_DIM[0]} {IMAGE_DIM[1]}",
            "--detector-size", f"{DETECTOR_SIZE_MM[0]} {DETECTOR_SIZE_MM[1]}",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            continue

        pfm = prefix + "0000.pfm"
        txt = prefix + "0000.txt"
        if not os.path.exists(pfm):
            found = glob.glob(prefix + "*.pfm")
            if found:
                pfm = found[0]
                txt = pfm.replace(".pfm", ".txt")

        if not os.path.exists(pfm) or not os.path.exists(txt):
            continue

        # Save PNG
        arr = read_pfm(pfm)
        png_path = os.path.join(view_dir, f"drr_{label}.png")
        save_png(arr, png_path)
        os.remove(pfm)

        # Save DCM
        dcm_path = os.path.join(view_dir, f"drr_{label}.dcm")
        try:
            save_dcm(png_path, dcm_path, ct_id, label)
        except Exception:
            pass  # dcm is optional, don't fail

        # Parse P matrix
        P = parse_plastimatch_txt(txt)
        os.remove(txt)

        # Save P
        np.savetxt(os.path.join(view_dir, f"P_{label}.txt"), P, fmt="%.8e")

        # Save geometry JSON
        geo = {
            "label": label,
            "angle_deg": float(angle_deg),
            "nrm": nrm.tolist(),
            "vup": [0.0, 0.0, 1.0],
            "sad_mm": SAD,
            "sid_mm": SID,
            "detector_size_mm": list(DETECTOR_SIZE_MM),
            "image_dim": list(IMAGE_DIM),
            "isocenter_mm": iso.tolist(),
            "source_mm": source.tolist(),
            "P": P.tolist(),
            "P_source": "plastimatch_native_parsed",
        }
        with open(os.path.join(view_dir, f"geometry_{label}.json"), 'w') as f:
            json.dump(geo, f, indent=2)

        ok += 1

    # Cleanup
    if os.path.exists(temp_ct):
        os.remove(temp_ct)

    return f"{ok}/10"


def main():
    print("=" * 70)
    print("  DRR GENERATION — ALL 1053 DATASETS")
    print("=" * 70)

    ct_dirs = sorted([d for d in glob.glob(os.path.join(ROOT, "*"))
                      if os.path.isdir(d) and os.path.exists(os.path.join(d, "ct.nii.gz"))])
    print(f"  Found {len(ct_dirs)} CT folders in {ROOT}\n")

    stats = {"skip": 0, "ok": 0, "fail": 0}
    t0 = time.time()

    for i, ct_dir in enumerate(ct_dirs):
        ct_id = os.path.basename(ct_dir)
        result = process_one_ct(ct_dir)

        if result == "skip":
            stats["skip"] += 1
            print(f"[{i+1}/{len(ct_dirs)}] SKIP  {ct_id}")
        elif result == "no_ct":
            stats["fail"] += 1
            print(f"[{i+1}/{len(ct_dirs)}] FAIL  {ct_id}  (no ct.nii.gz)")
        elif result == "not_lps":
            stats["fail"] += 1
            print(f"[{i+1}/{len(ct_dirs)}] FAIL  {ct_id}  (not LPS)")
        else:
            stats["ok"] += 1
            elapsed = time.time() - t0
            avg = elapsed / (stats["ok"] + stats["skip"])
            remaining = avg * (len(ct_dirs) - i - 1)
            print(f"[{i+1}/{len(ct_dirs)}] OK    {ct_id}  views={result}  "
                  f"ETA={remaining/60:.0f}min")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"  OK: {stats['ok']}  |  Skipped: {stats['skip']}  |  Failed: {stats['fail']}")
    print(f"  Next: python generate_mask_all.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
    
