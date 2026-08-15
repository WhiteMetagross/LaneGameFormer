#!/usr/bin/env python3
"""
Stratified Sampling Script for ChunkedProjectPrayagBEVDataset

This script creates a 20% stratified sample of the ChunkedProjectPrayagBEVDataset
for faster hyperparameter tuning while maintaining the distribution of:
- Scene IDs (DJI_0910, DJI_0911, DJI_0912, DJI_0914, DJI_0915)
- Time of day (morning, afternoon)
- Agent density (low, medium, high)
- Train/Val/Test split ratios

The output is a new dataset folder: StratifiedProjectPrayagBEVDataset

Author: LaneGameFormer Team
Date: December 2025
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATIO = 0.20  # 20% stratified sample
RANDOM_SEED = 42

# Source and destination paths
SOURCE_DIR = Path(r"c:\Users\Xeron\OneDrive\Documents\Programs\LaneGameFormer\ChunkedProjectPrayagBEVDataset10Hz")
DEST_DIR = Path(r"c:\Users\Xeron\OneDrive\Documents\Programs\LaneGameFormer\StratifiedProjectPrayagBEVDataset10Hz")


# ============================================================================
# Stratification Logic
# ============================================================================

def load_manifest(source_dir: Path) -> Dict:
    """Load the chunk manifest JSON file."""
    manifest_path = source_dir / "chunk_manifest.json"
    with open(manifest_path, 'r') as f:
        return json.load(f)


def create_strata_key(chunk: Dict) -> str:
    """
    Create a stratification key combining multiple attributes.
    
    Stratification dimensions:
    - original_scene_id: Ensures representation from all scenes
    - time_of_day: morning vs afternoon
    """
    scene = chunk['original_scene_id']
    time_of_day = chunk['time_of_day']
    
    return f"{scene}_{time_of_day}"


def stratified_sample(chunks: List[Dict], sample_ratio: float, seed: int) -> List[Dict]:
    """
    Perform stratified sampling on chunks.
    
    Groups chunks by strata key and samples proportionally from each stratum.
    Ensures at least 1 chunk per stratum if possible.
    """
    random.seed(seed)
    
    # Group chunks by strata
    strata = defaultdict(list)
    for chunk in chunks:
        key = create_strata_key(chunk)
        strata[key].append(chunk)
    
    sampled_chunks = []
    
    print(f"\nStratification breakdown:")
    print("-" * 60)
    
    for stratum_key, stratum_chunks in sorted(strata.items()):
        n_total = len(stratum_chunks)
        n_sample = max(1, round(n_total * sample_ratio))  # At least 1 per stratum
        
        # Random sample from this stratum
        sampled = random.sample(stratum_chunks, min(n_sample, n_total))
        sampled_chunks.extend(sampled)
        
        print(f"  {stratum_key}: {len(sampled)}/{n_total} chunks sampled")
    
    print("-" * 60)
    
    return sampled_chunks


def sample_split(split_chunks: List[Dict], sample_ratio: float, seed: int, split_name: str) -> List[Dict]:
    """Sample from a single split while maintaining stratification."""
    print(f"\n[{split_name.upper()}] Original: {len(split_chunks)} chunks")
    
    if len(split_chunks) == 0:
        return []
    
    sampled = stratified_sample(split_chunks, sample_ratio, seed)
    
    print(f"[{split_name.upper()}] Sampled: {len(sampled)} chunks ({100*len(sampled)/len(split_chunks):.1f}%)")
    
    return sampled


# ============================================================================
# File Operations
# ============================================================================

def get_chunk_files(chunk_id: str) -> List[str]:
    """Get all file patterns associated with a chunk."""
    return [
        f"{chunk_id}_tracks.csv",
        f"{chunk_id}_tracks.json",
        f"{chunk_id}_metadata.csv",
        f"{chunk_id}_road_annotation.json",
        f"{chunk_id}_road_mask.png",
        f"{chunk_id}_road_viz.png",
    ]


def copy_chunk_files(chunk: Dict, source_dir: Path, dest_dir: Path, split: str):
    """Copy all files for a chunk to the destination directory."""
    chunk_id = chunk['chunk_id']
    
    # Annotation files
    src_annot_dir = source_dir / split / "annotations"
    dst_annot_dir = dest_dir / split / "annotations"
    dst_annot_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in get_chunk_files(chunk_id):
        src_file = src_annot_dir / filename
        dst_file = dst_annot_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
    
    # Video file
    src_video_dir = source_dir / split / "videos"
    dst_video_dir = dest_dir / split / "videos"
    dst_video_dir.mkdir(parents=True, exist_ok=True)
    
    video_file = f"{chunk_id}.mp4"
    src_video = src_video_dir / video_file
    dst_video = dst_video_dir / video_file
    if src_video.exists():
        shutil.copy2(src_video, dst_video)


def copy_root_files(source_dir: Path, dest_dir: Path):
    """Copy root-level files (unified tracking data, etc.)."""
    root_files = [
        "unified_tracking_data.csv",
        "unified_tracking_data.json",
    ]
    
    for filename in root_files:
        src_file = source_dir / filename
        dst_file = dest_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_file)


# ============================================================================
# Manifest and Split File Generation
# ============================================================================

def create_sampled_manifest(
    original_manifest: Dict,
    train_chunks: List[Dict],
    val_chunks: List[Dict],
    test_chunks: List[Dict],
    dest_dir: Path
):
    """Create a new manifest for the sampled dataset."""
    
    # Update manifest with sampled chunks
    new_manifest = {
        "created": datetime.now().isoformat(),
        "source": "StratifiedProjectPrayagBEVDataset (20% sample of ChunkedProjectPrayagBEVDataset)",
        "sample_ratio": SAMPLE_RATIO,
        "random_seed": RANDOM_SEED,
        "parameters": original_manifest["parameters"],
        "statistics": {
            "total_chunks": len(train_chunks) + len(val_chunks) + len(test_chunks),
            "train_chunks": len(train_chunks),
            "val_chunks": len(val_chunks),
            "test_chunks": len(test_chunks),
        },
        "splits": {
            "train": train_chunks,
            "val": val_chunks,
            "test": test_chunks,
        },
        "folder_structure": original_manifest["folder_structure"],
        "original_statistics": original_manifest["statistics"],
    }
    
    # Write manifest
    manifest_path = dest_dir / "chunk_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(new_manifest, f, indent=2)
    
    return new_manifest


def create_split_files(
    train_chunks: List[Dict],
    val_chunks: List[Dict],
    test_chunks: List[Dict],
    dest_dir: Path
):
    """Create train_chunks.txt, val_chunks.txt, test_chunks.txt files."""
    
    for split_name, chunks in [("train", train_chunks), ("val", val_chunks), ("test", test_chunks)]:
        filepath = dest_dir / f"{split_name}_chunks.txt"
        chunk_ids = [c['chunk_id'] for c in chunks]
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(chunk_ids))


def create_readme(
    train_chunks: List[Dict],
    val_chunks: List[Dict],
    test_chunks: List[Dict],
    dest_dir: Path
):
    """Create a README documenting the stratified sample."""
    
    total = len(train_chunks) + len(val_chunks) + len(test_chunks)
    
    # Compute statistics
    all_chunks = train_chunks + val_chunks + test_chunks
    
    scene_dist = defaultdict(int)
    time_dist = defaultdict(int)
    density_dist = defaultdict(int)
    
    for chunk in all_chunks:
        scene_dist[chunk['original_scene_id']] += 1
        time_dist[chunk['time_of_day']] += 1
        density_dist[chunk['agent_density']] += 1
    
    readme_content = f"""# StratifiedProjectPrayagBEVDataset

## Overview

This is a **{SAMPLE_RATIO*100:.0f}% stratified sample** of the ChunkedProjectPrayagBEVDataset, 
designed for faster hyperparameter tuning while maintaining representative distribution.

## Sampling Methodology

- **Sample Ratio**: {SAMPLE_RATIO*100:.0f}%
- **Random Seed**: {RANDOM_SEED}
- **Stratification Dimensions**:
  - Scene ID (DJI_0910, DJI_0911, DJI_0912, DJI_0914, DJI_0915)
  - Time of Day (morning, afternoon)

## Dataset Statistics

### Split Distribution

| Split | Chunks | Percentage |
|-------|--------|------------|
| Train | {len(train_chunks)} | {100*len(train_chunks)/total:.1f}% |
| Validation | {len(val_chunks)} | {100*len(val_chunks)/total:.1f}% |
| Test | {len(test_chunks)} | {100*len(test_chunks)/total:.1f}% |
| **Total** | **{total}** | **100%** |

### Scene Distribution

| Scene ID | Chunks |
|----------|--------|
"""
    
    for scene, count in sorted(scene_dist.items()):
        readme_content += f"| {scene} | {count} |\n"
    
    readme_content += f"""
### Time of Day Distribution

| Time of Day | Chunks |
|-------------|--------|
"""
    
    for tod, count in sorted(time_dist.items()):
        readme_content += f"| {tod} | {count} |\n"
    
    readme_content += f"""
### Density Distribution

| Density | Chunks |
|---------|--------|
"""
    
    for density, count in sorted(density_dist.items()):
        readme_content += f"| {density} | {count} |\n"
    
    readme_content += f"""
## Usage

This dataset uses the same format as ChunkedProjectPrayagBEVDataset.
Update your config file to point to this directory:

```yaml
data:
  dataset_path: /path/to/StratifiedProjectPrayagBEVDataset
  use_chunked_dataset: true
```

## Chunks Included

### Train Chunks ({len(train_chunks)})
"""
    
    for chunk in sorted(train_chunks, key=lambda x: x['chunk_id']):
        readme_content += f"- {chunk['chunk_id']} ({chunk['time_of_day']}, {chunk['agent_density']})\n"
    
    readme_content += f"""
### Validation Chunks ({len(val_chunks)})
"""
    
    for chunk in sorted(val_chunks, key=lambda x: x['chunk_id']):
        readme_content += f"- {chunk['chunk_id']} ({chunk['time_of_day']}, {chunk['agent_density']})\n"
    
    readme_content += f"""
### Test Chunks ({len(test_chunks)})
"""
    
    for chunk in sorted(test_chunks, key=lambda x: x['chunk_id']):
        readme_content += f"- {chunk['chunk_id']} ({chunk['time_of_day']}, {chunk['agent_density']})\n"
    
    readme_content += f"""
## Generation Info

- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Source**: ChunkedProjectPrayagBEVDataset
- **Script**: create_stratified_dataset.py
"""
    
    readme_path = dest_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 70)
    print("Stratified Dataset Sampling for Hyperparameter Tuning")
    print("=" * 70)
    
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print(f"Sample Ratio: {SAMPLE_RATIO*100:.0f}%")
    print(f"Random Seed: {RANDOM_SEED}")
    
    # Load manifest
    print("\n[1/6] Loading chunk manifest...")
    manifest = load_manifest(SOURCE_DIR)
    
    original_train = manifest['splits']['train']
    original_val = manifest['splits']['val']
    original_test = manifest['splits']['test']
    
    print(f"  Original dataset: {len(original_train)} train, {len(original_val)} val, {len(original_test)} test")
    
    # Stratified sampling
    print("\n[2/6] Performing stratified sampling...")
    
    sampled_train = sample_split(original_train, SAMPLE_RATIO, RANDOM_SEED, "train")
    sampled_val = sample_split(original_val, SAMPLE_RATIO, RANDOM_SEED + 1, "val")
    sampled_test = sample_split(original_test, SAMPLE_RATIO, RANDOM_SEED + 2, "test")
    
    total_original = len(original_train) + len(original_val) + len(original_test)
    total_sampled = len(sampled_train) + len(sampled_val) + len(sampled_test)
    
    print(f"\nTotal: {total_sampled}/{total_original} chunks ({100*total_sampled/total_original:.1f}%)")
    
    # Create destination directory
    print("\n[3/6] Creating destination directory structure...")
    if DEST_DIR.exists():
        print(f"  Removing existing directory: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)
    
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    (DEST_DIR / "train" / "annotations").mkdir(parents=True, exist_ok=True)
    (DEST_DIR / "train" / "videos").mkdir(parents=True, exist_ok=True)
    (DEST_DIR / "val" / "annotations").mkdir(parents=True, exist_ok=True)
    (DEST_DIR / "val" / "videos").mkdir(parents=True, exist_ok=True)
    (DEST_DIR / "test" / "annotations").mkdir(parents=True, exist_ok=True)
    (DEST_DIR / "test" / "videos").mkdir(parents=True, exist_ok=True)
    
    # Copy files
    print("\n[4/6] Copying chunk files...")
    
    for i, chunk in enumerate(sampled_train):
        copy_chunk_files(chunk, SOURCE_DIR, DEST_DIR, "train")
        print(f"  Copied train chunk {i+1}/{len(sampled_train)}: {chunk['chunk_id']}")
    
    for i, chunk in enumerate(sampled_val):
        copy_chunk_files(chunk, SOURCE_DIR, DEST_DIR, "val")
        print(f"  Copied val chunk {i+1}/{len(sampled_val)}: {chunk['chunk_id']}")
    
    for i, chunk in enumerate(sampled_test):
        copy_chunk_files(chunk, SOURCE_DIR, DEST_DIR, "test")
        print(f"  Copied test chunk {i+1}/{len(sampled_test)}: {chunk['chunk_id']}")
    
    # Copy root files
    print("\n[5/6] Copying root-level files...")
    copy_root_files(SOURCE_DIR, DEST_DIR)
    
    # Create manifest and split files
    print("\n[6/6] Creating manifest and documentation...")
    create_sampled_manifest(manifest, sampled_train, sampled_val, sampled_test, DEST_DIR)
    create_split_files(sampled_train, sampled_val, sampled_test, DEST_DIR)
    create_readme(sampled_train, sampled_val, sampled_test, DEST_DIR)
    
    # Summary
    print("\n" + "=" * 70)
    print("STRATIFIED DATASET CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput Directory: {DEST_DIR}")
    print(f"\nDataset Summary:")
    print(f"  - Train chunks: {len(sampled_train)}")
    print(f"  - Val chunks: {len(sampled_val)}")
    print(f"  - Test chunks: {len(sampled_test)}")
    print(f"  - Total chunks: {total_sampled}")
    print(f"  - Sample ratio: {100*total_sampled/total_original:.1f}%")
    
    print("\nFiles created:")
    print(f"  - {DEST_DIR / 'chunk_manifest.json'}")
    print(f"  - {DEST_DIR / 'train_chunks.txt'}")
    print(f"  - {DEST_DIR / 'val_chunks.txt'}")
    print(f"  - {DEST_DIR / 'test_chunks.txt'}")
    print(f"  - {DEST_DIR / 'README.md'}")
    
    print("\nTo use this dataset for HPO, update your config:")
    print(f"  dataset_path: {DEST_DIR}")


if __name__ == "__main__":
    main()
