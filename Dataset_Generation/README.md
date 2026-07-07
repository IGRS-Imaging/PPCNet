## Dataset Generation

The complete dataset (1,037 patients with biplanar DRRs, projection matrices, and ground-truth point clouds) is available on HuggingFace:

**[PPCNet Dataset on HuggingFace](https://huggingface.co/datasets/ppcnet-dataset/PPCNet)**

The scripts in this folder reproduce the dataset generation pipeline from the source CT scans (VerSe + CTSpine1K):

1. `generate_drr.py` — Generates biplanar DRRs and projection matrices using Plastimatch.
2. `generate_gt_ppc.py` — Generates ground-truth point clouds.

# Generate DRRs
python generate_drr.py \
    --input_dir /path/to/Lumbar_ALL_LPS \
    --plastimatch /usr/bin/plastimatch

# Generate GT point clouds
python generate_gt_ppc.py \
    --input_dir /path/to/Lumbar_ALL_LPS \
    --output_dir /path/to/Lumbar_GT_PPC
