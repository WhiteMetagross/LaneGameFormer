#!/usr/bin/env python3
"""
chunk_interactions.py: Slices and maps global mined interaction JSON files to their respective
contiguous chunks for both 30Hz (ChunkedProjectPrayagBEVDataset) and 10Hz (ChunkedProjectPrayagBEVDataset10Hz).
Re-indexes frame numbers to be chunk-relative.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "ProjectPrayagTopDownDataset"
DST_30HZ_DIR = BASE_DIR / "ChunkedProjectPrayagBEVDataset"
DST_10HZ_DIR = BASE_DIR / "ChunkedProjectPrayagBEVDataset10Hz"
MANIFEST_PATH = DST_30HZ_DIR / "chunk_manifest.json"

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    # 1. Load global mined interactions
    print("Loading global mined interaction scenario databases...")
    
    conv_path = SRC_DIR / "conventional_ttc_scenarios.json"
    inter_path = SRC_DIR / "interaction_scenarios.json"
    novel_path = SRC_DIR / "novel_interaction_scenarios.json"
    
    with open(conv_path, 'r', encoding='utf-8') as f:
        global_conv = json.load(f)
    print(f"Loaded {len(global_conv)} conventional TTC scenarios.")
    
    with open(inter_path, 'r', encoding='utf-8') as f:
        global_inter = json.load(f)
    print(f"Loaded {len(global_inter)} interaction scenarios.")
    
    with open(novel_path, 'r', encoding='utf-8') as f:
        global_novel = json.load(f)
    print(f"Loaded {len(global_novel)} novel interaction scenarios.")

    # 2. Load chunk manifest blueprint
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    splits = manifest.get('splits', {})
    
    # Flatten all chunks
    chunks = []
    for split_name, chunk_list in splits.items():
        for chunk in chunk_list:
            chunk['split'] = split_name
            chunks.append(chunk)
            
    print(f"Processing interactions for {len(chunks)} chunks across both datasets...")
    
    # Tracking statistics for reporting
    stats = {}
    
    for chunk in tqdm(chunks):
        chunk_id = chunk['chunk_id']
        original_scene_id = chunk['original_scene_id']
        start_frame_0based = chunk['start_frame'] # 0-based
        end_frame_0based = chunk['end_frame']     # 0-based
        split = chunk['split']
        
        # 1-based original frame range for interactions
        chunk_start_1based = start_frame_0based + 1
        chunk_end_1based = end_frame_0based + 1
        
        # Slices for this chunk
        chunk_conv_30hz = []
        chunk_inter_30hz = []
        chunk_novel_30hz = []
        
        # Helper to shift to chunk-relative 1-based index: frame_id - start_frame_0based
        # (e.g. if chunk starts at 900, frame 901 becomes 901 - 900 = 1)
        
        # Conventional TTC
        for sc in global_conv:
            if sc['scene_id'] == original_scene_id:
                if sc['start_frame'] >= chunk_start_1based and sc['end_frame'] <= chunk_end_1based:
                    new_sc = sc.copy()
                    new_sc['start_frame'] = sc['start_frame'] - start_frame_0based
                    new_sc['end_frame'] = sc['end_frame'] - start_frame_0based
                    # Rename scene_id to chunk_id to align with chunk tracks context
                    new_sc['scene_id'] = chunk_id
                    chunk_conv_30hz.append(new_sc)
                    
        # Interaction Scenarios
        for sc in global_inter:
            if sc['scene_id'] == original_scene_id:
                if sc['start_frame'] >= chunk_start_1based and sc['end_frame'] <= chunk_end_1based:
                    new_sc = sc.copy()
                    new_sc['start_frame'] = sc['start_frame'] - start_frame_0based
                    new_sc['end_frame'] = sc['end_frame'] - start_frame_0based
                    new_sc['scene_id'] = chunk_id
                    chunk_inter_30hz.append(new_sc)
                    
        # Novel Interaction Scenarios
        for sc in global_novel:
            if sc['scene_id'] == original_scene_id:
                if sc['start_frame'] >= chunk_start_1based and sc['end_frame'] <= chunk_end_1based:
                    new_sc = sc.copy()
                    new_sc['start_frame'] = sc['start_frame'] - start_frame_0based
                    new_sc['end_frame'] = sc['end_frame'] - start_frame_0based
                    new_sc['scene_id'] = chunk_id
                    chunk_novel_30hz.append(new_sc)
                    
        # Define directories for 30Hz & 10Hz
        annot_30hz_dir = DST_30HZ_DIR / split / "annotations"
        annot_10hz_dir = DST_10HZ_DIR / split / "annotations"
        
        ensure_dir(annot_30hz_dir)
        ensure_dir(annot_10hz_dir)
        
        # Write 30Hz JSONs
        with open(annot_30hz_dir / f"{chunk_id}_conventional_ttc_scenarios.json", 'w', encoding='utf-8') as f:
            json.dump(chunk_conv_30hz, f, indent=4)
        with open(annot_30hz_dir / f"{chunk_id}_interaction_scenarios.json", 'w', encoding='utf-8') as f:
            json.dump(chunk_inter_30hz, f, indent=4)
        with open(annot_30hz_dir / f"{chunk_id}_novel_interaction_scenarios.json", 'w', encoding='utf-8') as f:
            json.dump(chunk_novel_30hz, f, indent=4)
            
        # Create 10Hz versions using formula: new_frame = (frame - 1) // 3 + 1
        chunk_conv_10hz = []
        for sc in chunk_conv_30hz:
            new_sc = sc.copy()
            new_sc['start_frame'] = (sc['start_frame'] - 1) // 3 + 1
            new_sc['end_frame'] = (sc['end_frame'] - 1) // 3 + 1
            chunk_conv_10hz.append(new_sc)
            
        chunk_inter_10hz = []
        for sc in chunk_inter_30hz:
            new_sc = sc.copy()
            new_sc['start_frame'] = (sc['start_frame'] - 1) // 3 + 1
            new_sc['end_frame'] = (sc['end_frame'] - 1) // 3 + 1
            chunk_inter_10hz.append(new_sc)
            
        chunk_novel_10hz = []
        for sc in chunk_novel_30hz:
            new_sc = sc.copy()
            new_sc['start_frame'] = (sc['start_frame'] - 1) // 3 + 1
            new_sc['end_frame'] = (sc['end_frame'] - 1) // 3 + 1
            chunk_novel_10hz.append(new_sc)
            
        # Write 10Hz JSONs
        with open(annot_10hz_dir / f"{chunk_id}_conventional_ttc_scenarios.json", 'w', encoding='utf-8') as f:
            json.dump(chunk_conv_10hz, f, indent=4)
        with open(annot_10hz_dir / f"{chunk_id}_interaction_scenarios.json", 'w', encoding='utf-8') as f:
            json.dump(chunk_inter_10hz, f, indent=4)
        with open(annot_10hz_dir / f"{chunk_id}_novel_interaction_scenarios.json", 'w', encoding='utf-8') as f:
            json.dump(chunk_novel_10hz, f, indent=4)
            
        # Record stats for reporting
        stats[chunk_id] = {
            "conventional_count": len(chunk_conv_30hz),
            "interaction_count": len(chunk_inter_30hz),
            "novel_count": len(chunk_novel_30hz)
        }
        
    # Summarize overall numbers
    total_conv = sum(s["conventional_count"] for s in stats.values())
    total_inter = sum(s["interaction_count"] for s in stats.values())
    total_novel = sum(s["novel_count"] for s in stats.values())
    
    print("\n" + "="*40)
    print("INTERACTION SLICING SUMMARY")
    print("="*40)
    print(f"Total sliced Conventional TTC Scenarios: {total_conv}")
    print(f"Total sliced Interaction Scenarios: {total_inter}")
    print(f"Total sliced Novel Interaction Scenarios: {total_novel}")
    print(f"Average scenarios per chunk: Conventional={total_conv/len(chunks):.1f}, Interaction={total_inter/len(chunks):.1f}, Novel={total_novel/len(chunks):.1f}")
    print("="*40)
    
    # Save a stats file for easy reference
    with open(DST_30HZ_DIR / "chunk_interactions_summary.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    with open(DST_10HZ_DIR / "chunk_interactions_summary.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
        
    print("Interaction slicing and downsampling completed successfully!")

if __name__ == "__main__":
    main()
