# ============================================================
# Run locally: sample and save a subset of participant embeddings
# ============================================================
import numpy as np
import os
import shutil

PARTICIPANT_EMB_DIR = os.path.expanduser(
    "~/Documents/UCI/CCNL/HumanEmbedding/participant_embeddings/"
)
OUTPUT_DIR = os.path.expanduser(
    "~/Documents/UCI/CCNL/HumanEmbedding/participant_embeddings_subset/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SAMPLE = 100   # adjust — 500 × ~3MB each ≈ 1.5GB, comfortable for Colab upload
rng      = np.random.default_rng(42)

all_files  = sorted(os.listdir(PARTICIPANT_EMB_DIR))
sampled    = rng.choice(all_files, size=min(N_SAMPLE, len(all_files)), replace=False)

print(f"Sampling {len(sampled)} / {len(all_files)} participants...")

for i, fname in enumerate(sampled):
    src = os.path.join(PARTICIPANT_EMB_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    shutil.copy2(src, dst)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(sampled)}")

# Check total size
total_bytes = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
)
print(f"\nDone. {len(sampled)} files saved to {OUTPUT_DIR}")
print(f"Total size: {total_bytes / 1e9:.2f} GB")