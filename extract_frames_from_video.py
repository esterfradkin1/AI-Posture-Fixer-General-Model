# ============================================================
# STEP 1 – VIDEO TO FRAME EXTRACTION
# ============================================================
#
# Purpose:
#   Convert a short video into a sequence of image frames
#   sampled at a lower frame rate for manual labeling and
#   dataset creation.
#
# Input:
#   - video_path:
#       Path to a recorded video containing posture footage.
#
# Processing:
#   - Reads the video using OpenCV.
#   - Detects the original video FPS.
#   - Saves only every Nth frame to downsample the video
#     to approximately 5 FPS.
#
# Output:
#   - output_folder:
#       A directory containing extracted image frames
#       (e.g., frame_0000.jpg, frame_0001.jpg, ...).
#
# Notes:
#   - Downsampling reduces redundancy between frames.
#   - The extracted frames are later labeled in CVAT (Step 2)
#     and used for feature extraction (Step 3).
# ============================================================

import cv2
import os
from pathlib import Path

from openpyxl.styles.builtins import output


def activate(video_path, output_folder,id):
    # 1. Setup paths
    video_path = Path(video_path)#Path to input video
    output_folder = Path(output_folder)/f"Output_{video_path.stem}"# Folder for extracted frames
    output_folder.mkdir(parents=True, exist_ok=True)

    # 2. Open the video
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)   # Original FPS of the video
    hop_interval = max(1,int(video_fps / 5))       # Save frames at ~5 FPS

    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Save only every Nth frame to reduce frame rate
        if count % hop_interval == 0:
            filename = Path(output_folder) / f"{video_path.stem}_frame_{saved_count}.jpg"
            cv2.imwrite(str(filename), frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Done! Saved {saved_count} images to '{output_folder}'.")