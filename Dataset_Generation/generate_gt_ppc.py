import vtk
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree
from skimage import measure
import nibabel as nib
import time
import csv
import argparse

# ─── CONFIG ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate ground-truth point clouds")
parser.add_argument("--input_dir", type=str, required=True,
                    help="Root directory containing patient folders with ct.nii.gz and seg.nii.gz")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for GT point clouds")
parser.add_argument("--num_patients", type=int, default=1053,
                    help="Number of patients (default: 1053)")
parser.add_argument("--num_points", type=int, default=5120,
                    help="Points per patient (default: 5120)")
args = parser.parse_args()

DATASET_ROOT = Path(args.input_dir)
OUTPUT_ROOT  = Path(args.output_dir)

# All patients in dataset
PATIENT_IDS = [f"lumbar_{i:04d}" for i in range(1, args.num_patients + 1)]

# VerSe label mapping
LEVELS_MAP = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5"}

# GT PPC settings
N_GT_POINTS    = args.num_points      # Total points per patient
MIN_GT_VOXELS  = 5000      # Skip patients with degenerate labels
ICP_THRESHOLD  = 50.0      # mm for rigid alignment

# Retry settings for filesystem robustness
MAX_RETRIES    = 5
RETRY_DELAY    = 3         # seconds (increases per retry)

# Minimum valid file size (skip-if-done check)
MIN_VALID_VTK_SIZE = 1000  # bytes — sanity check that file isn't truncated
# ──────────────────────────────────────────────────────────

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════
# I/O helpers with retry
# ═════════════════════════════════════════════════════════

def load_ply_points(ply_path):
    """Load .ply as Nx3 numpy array with retry."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            reader = vtk.vtkPLYReader()
            reader.SetFileName(str(ply_path))
            reader.Update()
            ppc = reader.GetOutput()
            n_pts = ppc.GetNumberOfPoints()
            if n_pts == 0:
                raise ValueError("PLY file empty or unreadable")
            return np.array([ppc.GetPoint(i) for i in range(n_pts)])
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    PLY retry {attempt+1}/{MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)
    raise last_error


def load_nifti(seg_path):
    """Load NIfTI file with retry."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            img = nib.load(seg_path)
            data = img.get_fdata()
            return img, data
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    NIfTI retry {attempt+1}/{MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)
    raise last_error


def save_vtk_points(points, path):
    """Save Nx3 array as .vtk with vertex cells. Atomic write via temp file."""
    # Write to temp file first, then rename — prevents corrupted partials
    temp_path = path.with_suffix(".vtk.tmp")
    
    vtk_points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    for i, pt in enumerate(points):
        vtk_points.InsertNextPoint(pt)
        cells.InsertNextCell(1)
        cells.InsertCellPoint(i)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetVerts(cells)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(str(temp_path))
    writer.SetInputData(poly)
    writer.SetFileTypeToBinary()
    writer.Write()
    
    # Atomic rename — file appears only once fully written
    if path.exists():
        path.unlink()
    temp_path.rename(path)


# ═════════════════════════════════════════════════════════
# Processing functions
# ═════════════════════════════════════════════════════════

def check_gt_quality(seg_path, min_voxels=MIN_GT_VOXELS):
    """Return {level: (is_valid, n_voxels)} dict."""
    img, data = load_nifti(seg_path)
    quality = {}
    for label_val, level in LEVELS_MAP.items():
        n_voxels = int(np.sum(data == label_val))
        quality[level] = (n_voxels >= min_voxels, n_voxels)
    return quality


def extract_vertebra_surface(seg_path, label_val):
    """Marching cubes for one vertebra, returns surface points in CT space."""
    img, data = load_nifti(seg_path)
    spacing = img.header.get_zooms()
    mask = (data == label_val).astype(np.uint8)
    if mask.sum() == 0:
        return None
    verts, _, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
    affine = img.affine
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    return (affine @ verts_h.T).T[:, :3]


def align_ct_to_projection_space(ct_points, projection_points, threshold=ICP_THRESHOLD):
    """Rigid ICP with 4 flip initializations to handle coordinate flips."""
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(ct_points)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(projection_points)
    
    src_c = ct_points.mean(axis=0)
    tgt_c = projection_points.mean(axis=0)
    
    best = None
    best_fit = -1
    for flip in [(1,1,1), (-1,1,1), (1,-1,1), (-1,-1,1)]:
        T = np.eye(4)
        T[0,0], T[1,1], T[2,2] = flip
        T[:3, 3] = tgt_c - T[:3,:3] @ src_c
        r = o3d.pipelines.registration.registration_icp(
            src, tgt, threshold, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        if r.fitness > best_fit:
            best_fit = r.fitness
            best = r
    
    src.transform(best.transformation)
    return np.asarray(src.points), best.fitness


def sample_uniform_from_points(points, n_target):
    """Farthest-point sampling for uniform coverage."""
    if len(points) <= n_target:
        extra = n_target - len(points)
        idx = np.random.choice(len(points), extra, replace=True)
        return np.vstack([points, points[idx]])
    
    selected = [np.random.randint(len(points))]
    distances = np.full(len(points), np.inf)
    
    for _ in range(n_target - 1):
        last_idx = selected[-1]
        d = np.linalg.norm(points - points[last_idx], axis=1)
        distances = np.minimum(distances, d)
        next_idx = int(np.argmax(distances))
        selected.append(next_idx)
    
    return points[selected]


def is_already_done(patient_id):
    """Check if patient's gt_ppc.vtk already exists and is valid."""
    out_path = OUTPUT_ROOT / patient_id / "gt_ppc.vtk"
    if not out_path.exists():
        return False
    if out_path.stat().st_size < MIN_VALID_VTK_SIZE:
        return False
    return True


# ═════════════════════════════════════════════════════════
# Load existing log to preserve history across runs
# ═════════════════════════════════════════════════════════

log_path = OUTPUT_ROOT / "processing_log.csv"
existing_log = {}
if log_path.exists():
    try:
        with open(log_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 5:
                    existing_log[row[0]] = row
    except Exception as e:
        print(f"Note: couldn't read existing log: {e}")


# ═════════════════════════════════════════════════════════
# Pre-scan: count already-completed patients
# ═════════════════════════════════════════════════════════

print("="*60)
print("GT PPC DATASET GENERATION (resumable)")
print("="*60)
print(f"Dataset: {DATASET_ROOT}")
print(f"Output:  {OUTPUT_ROOT}")
print(f"Target points per patient: {N_GT_POINTS}")
print(f"Total patients in list: {len(PATIENT_IDS)}")

print("\nScanning for already-completed patients...")
already_done_count = sum(1 for pid in PATIENT_IDS if is_already_done(pid))
remaining = len(PATIENT_IDS) - already_done_count
print(f"  ✓ Already done: {already_done_count}")
print(f"  → Remaining:    {remaining}")
print()


# ═════════════════════════════════════════════════════════
# Main processing loop
# ═════════════════════════════════════════════════════════

t_start = time.time()
log_rows = list(existing_log.values())  # preserve previous run's log
log_keys = set(existing_log.keys())

success_count = 0
skip_count = 0
fail_count = 0
resume_count = 0

for p_idx, patient_id in enumerate(PATIENT_IDS):
    elapsed_min = (time.time() - t_start) / 60
    
    # ── SKIP-IF-DONE check (resume capability)
    if is_already_done(patient_id):
        resume_count += 1
        if (p_idx + 1) % 100 == 0 or (p_idx + 1) == len(PATIENT_IDS):
            print(f"[{p_idx+1}/{len(PATIENT_IDS)}] checkpoint: "
                  f"{resume_count} skipped (already done), "
                  f"{success_count} new completions, "
                  f"{elapsed_min:.1f} min elapsed")
        continue
    
    print(f"[{p_idx+1}/{len(PATIENT_IDS)}] {patient_id}  "
          f"({elapsed_min:.1f} min elapsed)")
    
    patient_dir = DATASET_ROOT / patient_id
    ply_path = patient_dir / "AP_0" / "filled_point_cloud.ply"
    seg_path = patient_dir / "seg.nii.gz"
    
    # ── Check input files exist
    if not ply_path.exists():
        print(f"  ✗ Missing PPC")
        log_rows.append([patient_id, "missing_ppc", 0, 0, 0.0])
        log_keys.add(patient_id)
        fail_count += 1
        continue
    if not seg_path.exists():
        print(f"  ✗ Missing seg")
        log_rows.append([patient_id, "missing_seg", 0, 0, 0.0])
        log_keys.add(patient_id)
        fail_count += 1
        continue
    
    # ── Check GT quality
    try:
        quality = check_gt_quality(seg_path)
    except Exception as e:
        print(f"  ✗ Failed to read seg: {e}")
        log_rows.append([patient_id, "seg_read_error", 0, 0, 0.0])
        log_keys.add(patient_id)
        fail_count += 1
        continue
    
    valid_levels = [(lv, n) for lv, (ok, n) in quality.items() if ok]
    invalid_levels = [(lv, n) for lv, (ok, n) in quality.items() if not ok]
    
    if len(valid_levels) < 3:
        print(f"  ⚠ Too few valid vertebrae: {valid_levels} — skipping")
        log_rows.append([patient_id, "too_few_valid_vertebrae",
                         len(valid_levels), 0, 0.0])
        log_keys.add(patient_id)
        skip_count += 1
        continue
    
    if invalid_levels:
        print(f"  ⚠ Degenerate: {invalid_levels}")
    
    # ── Load projection-space reference
    try:
        projection_points = load_ply_points(ply_path)
    except Exception as e:
        print(f"  ✗ Failed to load PPC after retries: {e}")
        log_rows.append([patient_id, "ppc_load_error", 0, 0, 0.0])
        log_keys.add(patient_id)
        fail_count += 1
        continue
    
    # ── Extract surface for valid vertebrae
    ct_surfaces = []
    for label_val, level in LEVELS_MAP.items():
        if not quality[level][0]:
            continue
        try:
            surf = extract_vertebra_surface(seg_path, label_val)
            if surf is not None and len(surf) > 100:
                ct_surfaces.append(surf)
        except Exception as e:
            print(f"  ⚠ {level} marching cubes failed: {e}")
    
    if not ct_surfaces:
        print(f"  ✗ No surfaces extracted")
        log_rows.append([patient_id, "no_surfaces", 0, 0, 0.0])
        log_keys.add(patient_id)
        fail_count += 1
        continue
    
    ct_full_spine = np.vstack(ct_surfaces)
    
    # ── Align CT-space spine to projection-space dense PPC
    try:
        aligned_spine, fitness = align_ct_to_projection_space(
            ct_full_spine, projection_points)
    except Exception as e:
        print(f"  ✗ ICP failed: {e}")
        log_rows.append([patient_id, "icp_error", 0, 0, 0.0])
        log_keys.add(patient_id)
        fail_count += 1
        continue
    
    if fitness < 0.3:
        print(f"  ⚠ Low ICP fitness ({fitness:.2f}) — alignment unreliable")
    
    # ── Sample uniformly
    gt_ppc = sample_uniform_from_points(aligned_spine, N_GT_POINTS)
    
    # ── Save (atomic write via temp file)
    out_dir = OUTPUT_ROOT / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gt_ppc.vtk"
    
    try:
        save_vtk_points(gt_ppc, out_path)
    except Exception as e:
        print(f"  ✗ Save failed: {e}")
        log_rows.append([patient_id, "save_error", 0, 0, 0.0])
        log_keys.add(patient_id)
        fail_count += 1
        continue
    
    print(f"  ✓ Saved: {out_path.name}  "
          f"({len(gt_ppc)} pts, ICP fitness {fitness:.3f}, "
          f"{len(valid_levels)} valid vertebrae)")
    
    # Update log (remove old entry if rerunning, append new)
    if patient_id in log_keys:
        log_rows = [row for row in log_rows if row[0] != patient_id]
    log_rows.append([patient_id, "ok", len(valid_levels),
                     len(gt_ppc), fitness])
    log_keys.add(patient_id)
    success_count += 1
    
    # ── Save log every 25 patients (so progress isn't lost)
    if success_count % 25 == 0:
        try:
            with open(log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["patient_id", "status", "n_valid_vertebrae",
                            "n_points_saved", "icp_fitness"])
                w.writerows(log_rows)
        except Exception as e:
            print(f"  Note: log save failed (will retry): {e}")


# ═════════════════════════════════════════════════════════
# Final summary and log
# ═════════════════════════════════════════════════════════

total_time = (time.time() - t_start) / 60
print(f"\n{'='*60}")
print("DATASET GENERATION COMPLETE")
print(f"{'='*60}")
print(f"Total time this run: {total_time:.1f} min")
print(f"  ✓ Newly completed:  {success_count}")
print(f"  ⏭ Resumed (skipped): {resume_count}")
print(f"  ⚠ Skipped (bad GT):  {skip_count}")
print(f"  ✗ Failed:            {fail_count}")
print(f"  ─────────────────────────")
total_done = success_count + resume_count
print(f"  Total GT PPCs ready: {total_done} / {len(PATIENT_IDS)}")

# Save final log
try:
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "status", "n_valid_vertebrae",
                    "n_points_saved", "icp_fitness"])
        w.writerows(log_rows)
    print(f"\nLog saved: {log_path}")
except Exception as e:
    print(f"Final log save failed: {e}")

print(f"Output folder: {OUTPUT_ROOT}")
