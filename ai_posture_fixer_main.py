############ Main - AI Posture Fixer ##############

from pathlib import Path

import extract_frames_from_video

"""
Extract frames from video - handling with creating frames from video
for all videos we have.
"""

if __name__ == '__main__':

    # Folder that contains all input videos
    main_all_input_videos_path = Path(r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames"
                                      r"\pythonProject1\extract_frames_from_video-All_input_videos\new_videos")

    # Supported video formats
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg"}

    # Check folder exists
    if not main_all_input_videos_path.exists():
        print(f"[ERROR] Folder does not exist:\n{main_all_input_videos_path}")
        exit()

    # Create list of video paths
    video_paths = [
        file for file in main_all_input_videos_path.iterdir()
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
    ]

    # Print results
    for path in video_paths:
        print(path)
        name = Path(path.stem)
        print(name)

    print(f"\nTotal videos found: {len(video_paths)}")

    output_folder = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
                     r"\extract_frames_from_video-All_output_frames")

    for path in video_paths:
        id = Path(path.stem)
        print("id=" + str(id))
        extract_frames_from_video.activate(path,output_folder,id)
