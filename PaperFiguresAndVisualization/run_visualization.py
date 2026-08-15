#This file runs the visualization process for all tracked scenes.

import os
import glob
import json
import config
import visualizer

#This function runs visualization for all processed scenes.
def run():
    print("Doing Visualization:")
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)

    if not os.path.exists(config.INTERMEDIATE_DIR):
        print(f"Error: Intermediate directory not found at '{config.INTERMEDIATE_DIR}'.")
        print("Run the 'python main.py' first.")
        return

    scene_dirs = [d for d in glob.glob(os.path.join(config.INTERMEDIATE_DIR, '*')) if os.path.isdir(d)]

    if not scene_dirs:
        print("No processed scenes found to visualize.")
        return

    print(f"Found {len(scene_dirs)} processed scene(s).")

    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir)
        print(f"\nVisualizing Scene: {scene_id}")

        video_path = None
        for fmt in config.SUPPORTED_VIDEO_FORMATS:
            path_attempt = os.path.join(config.INPUT_DIR, f"{scene_id}{fmt}")
            if os.path.exists(path_attempt):
                video_path = path_attempt
                break

        # Fallback to temp_videos
        if not video_path:
            for fmt in config.SUPPORTED_VIDEO_FORMATS:
                path_attempt = os.path.join(config.BASE_DIR, "temp_videos", f"{scene_id}{fmt}")
                if os.path.exists(path_attempt):
                    video_path = path_attempt
                    break

        if not video_path:
            print(f"  Warning: Original video for '{scene_id}' not found. Skipping.")
            continue

        tracks_path = os.path.join(scene_dir, f"{scene_id}_tracks.json")
        if not os.path.exists(tracks_path):
            print(f"  Warning: Tracking data for '{scene_id}' not found. Skipping.")
            continue

        visualizer.visualize_tracks_for_scene(video_path, tracks_path, scene_id)

    print("\nVisualization complete.")

if __name__ == "__main__":
    run()