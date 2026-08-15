"""
Dataset Cleanup Script

Deletes unused files from Prayag BEV datasets that are not needed for HPO/Training:
- Video files (.mp4) in */videos/ folders
- JSON tracks (*_tracks.json)
- Metadata CSV (*_metadata.csv)
- Visualization PNG (*_road_viz.png)
- Road mask PNG (*_road_mask.png)
- Unified tracking data (unified_tracking_data.*)
- Pre-built graphs (graphs/ folder)

Files KEPT (required for training):
- *_tracks.csv (trajectory data)
- *_road_annotation.json (road/lane annotations)
- train_chunks.txt, val_chunks.txt, test_chunks.txt
- chunk_manifest.json
"""

import os
import shutil
import stat
from pathlib import Path


def remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree to handle read-only files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def force_delete_file(file_path: Path) -> bool:
    """Force delete a file, removing read-only attribute if needed."""
    try:
        if file_path.exists():
            # Remove read-only attribute
            os.chmod(file_path, stat.S_IWRITE)
            file_path.unlink()
            return True
    except PermissionError as e:
        print(f"  [ERROR] Permission denied: {file_path}")
        print(f"          {e}")
    except Exception as e:
        print(f"  [ERROR] Failed to delete {file_path}: {e}")
    return False


def force_delete_folder(folder_path: Path) -> bool:
    """Force delete a folder, removing read-only attributes if needed."""
    try:
        if folder_path.exists():
            shutil.rmtree(folder_path, onerror=remove_readonly)
            return True
    except PermissionError as e:
        print(f"  [ERROR] Permission denied: {folder_path}")
        print(f"          {e}")
    except Exception as e:
        print(f"  [ERROR] Failed to delete {folder_path}: {e}")
    return False


def get_size_str(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} bytes"


def get_folder_size(folder_path: Path) -> int:
    """Calculate total size of a folder."""
    total = 0
    try:
        for item in folder_path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except:
        pass
    return total


def cleanup_dataset(dataset_path: Path) -> dict:
    """Clean up a single dataset folder."""
    stats = {
        "files_deleted": 0,
        "folders_deleted": 0,
        "bytes_freed": 0,
        "errors": 0
    }
    
    if not dataset_path.exists():
        print(f"  [SKIP] Dataset not found: {dataset_path}")
        return stats
    
    print(f"\n{'='*60}")
    print(f"Cleaning: {dataset_path.name}")
    print(f"{'='*60}")
    
    # 1. Delete unified_tracking_data.* files
    print("\n[1/4] Deleting unified_tracking_data files...")
    for pattern in ["unified_tracking_data.csv", "unified_tracking_data.json"]:
        file_path = dataset_path / pattern
        if file_path.exists():
            size = file_path.stat().st_size
            if force_delete_file(file_path):
                print(f"  Deleted: {pattern} ({get_size_str(size)})")
                stats["files_deleted"] += 1
                stats["bytes_freed"] += size
            else:
                stats["errors"] += 1
        else:
            print(f"  [OK] Already deleted: {pattern}")
    
    # 2. Delete graphs/ folder
    print("\n[2/4] Deleting graphs folder...")
    graphs_path = dataset_path / "graphs"
    if graphs_path.exists():
        size = get_folder_size(graphs_path)
        if force_delete_folder(graphs_path):
            print(f"  Deleted: graphs/ ({get_size_str(size)})")
            stats["folders_deleted"] += 1
            stats["bytes_freed"] += size
        else:
            stats["errors"] += 1
    else:
        print(f"  [OK] Already deleted: graphs/")
    
    # 3. Delete video folders
    print("\n[3/4] Deleting video folders...")
    for split in ["train", "val", "test"]:
        videos_path = dataset_path / split / "videos"
        if videos_path.exists():
            size = get_folder_size(videos_path)
            if force_delete_folder(videos_path):
                print(f"  Deleted: {split}/videos/ ({get_size_str(size)})")
                stats["folders_deleted"] += 1
                stats["bytes_freed"] += size
            else:
                stats["errors"] += 1
        else:
            print(f"  [OK] Already deleted: {split}/videos/")
    
    # 4. Delete unused annotation files
    print("\n[4/4] Deleting unused annotation files...")
    patterns_to_delete = [
        "*_tracks.json",      # JSON tracks (keep CSV)
        "*_metadata.csv",     # Metadata
        "*_road_viz.png",     # Visualization
        "*_road_mask.png"     # Road mask (keep road_annotation.json)
    ]
    
    for split in ["train", "val", "test"]:
        ann_path = dataset_path / split / "annotations"
        if not ann_path.exists():
            continue
        
        for pattern in patterns_to_delete:
            files = list(ann_path.glob(pattern))
            if files:
                total_size = sum(f.stat().st_size for f in files if f.exists())
                deleted_count = 0
                for file_path in files:
                    if force_delete_file(file_path):
                        deleted_count += 1
                    else:
                        stats["errors"] += 1
                
                if deleted_count > 0:
                    print(f"  Deleted: {split}/annotations/{pattern} ({deleted_count} files, {get_size_str(total_size)})")
                    stats["files_deleted"] += deleted_count
                    stats["bytes_freed"] += total_size
    
    return stats


def main():
    """Main function to clean up all datasets."""
    print("="*70)
    print("DATASET CLEANUP SCRIPT")
    print("="*70)
    print("\nThis script will delete unused files from Prayag BEV datasets.")
    print("Files to delete: videos, *_tracks.json, *_metadata.csv,")
    print("                 *_road_viz.png, *_road_mask.png, unified_tracking_data.*")
    print("\nFiles KEPT: *_tracks.csv, *_road_annotation.json, chunk lists, manifest")
    
    # Dataset paths
    base_path = Path(r"C:\Users\Xeron\OneDrive\Documents\LargeDatasets")
    
    datasets = [
        base_path / "ChunkedProjectPrayagBEVDataset",
        base_path / "ChunkedProjectPrayagBEVDataset10Hz",
        base_path / "StratifiedProjectPrayagBEVDataset",
        base_path / "StratifiedProjectPrayagBEVDataset10Hz"
    ]
    
    # Check which datasets exist
    print(f"\nBase path: {base_path}")
    print("\nDatasets found:")
    for ds in datasets:
        status = "✓ EXISTS" if ds.exists() else "✗ NOT FOUND"
        print(f"  {status}: {ds.name}")
    
    # Confirm
    print("\n" + "-"*70)
    response = input("Proceed with cleanup? (yes/no): ").strip().lower()
    if response != "yes":
        print("Cleanup cancelled.")
        return
    
    # Clean each dataset
    total_stats = {
        "files_deleted": 0,
        "folders_deleted": 0,
        "bytes_freed": 0,
        "errors": 0
    }
    
    for dataset_path in datasets:
        stats = cleanup_dataset(dataset_path)
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Summary
    print("\n" + "="*70)
    print("CLEANUP COMPLETE")
    print("="*70)
    print(f"\nTotal files deleted:   {total_stats['files_deleted']}")
    print(f"Total folders deleted: {total_stats['folders_deleted']}")
    print(f"Total space freed:     {get_size_str(total_stats['bytes_freed'])}")
    print(f"Errors encountered:    {total_stats['errors']}")
    
    if total_stats['errors'] > 0:
        print("\n[WARNING] Some files could not be deleted.")
        print("Try running this script as Administrator, or")
        print("pause OneDrive sync before running.")


if __name__ == "__main__":
    main()
