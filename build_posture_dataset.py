# ============================================================
# STEP 3 – FEATURE EXTRACTION + DATASET + 10 PREVIEW IMAGES
# ============================================================
# Purpose:
#   Generate a structured machine-learning dataset from labeled
#   posture images.
#
# Inputs:
#   - IMAGE_DIR:
#       Directory containing posture images (frames extracted from video).
#   - JSON_PATH:
#       CVAT-exported JSON file containing posture labels per image.
#
# Processing:
#   - Runs MediaPipe Pose on each image to detect body landmarks.
#   - Extracts posture-related geometric features (head, shoulders, torso).
#   - If a required landmark is missing, the corresponding feature
#     is set to NaN (not zero).
#   - Raw landmark coordinates and visibility scores are always stored.
#   - Matches each image with its corresponding label from the JSON file.
#   - Saves annotated images with pose landmarks for visual verification.
#
# Outputs:
#   - OUTPUT_FILE (Excel):
#       A dataset where each row corresponds to one image and includes:
#         • Image ID
#         • Posture label
#         • Extracted posture features
#         • Raw landmark coordinates and visibility scores
#   - OUTPUT_LANDMARK_DIR:
#       Folder containing images with detected landmarks drawn.
#
# Notes:
#   - Designed for offline preprocessing only.
#   - Uses very low detection thresholds to capture subtle posture variations.
#   - Missing features are handled later via median imputation
#     during model training and live inference.
# ============================================================
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
import re  # Add this at the top with your other imports

# =========================
# MediaPipe setup
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.01,
    min_tracking_confidence=0.01
)

# =========================
# Geometry helpers
# =========================

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def wrap_to_pi(angle):
    """
    Wrap an angle (radians) to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
def extract_math_features(landmarks):
    """
    Extract posture-related geometric features and raw landmark data.

    Missing landmarks → feature = NaN.
    """

    def get_pt(idx):
        """
        Return landmark only if visibility is sufficient.
        """
        pt = landmarks[idx.value]
        return pt if pt.visibility > 0.1 else None

    def add_landmark_fields(res, prefix, lm):
        """
        Store raw landmark coordinates and visibility.
        """
        res[f"{prefix}_x"] = np.nan if lm is None else float(lm.x)
        res[f"{prefix}_y"] = np.nan if lm is None else float(lm.y)
        res[f"{prefix}_z"] = np.nan if lm is None else float(lm.z)
        res[f"{prefix}_vis"] = np.nan if lm is None else float(lm.visibility)

    # --------------------------------------------------------
    # Key landmarks
    # --------------------------------------------------------
    LS   = get_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    RS   = get_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    NOSE = get_pt(mp_pose.PoseLandmark.NOSE)
    LEAR = get_pt(mp_pose.PoseLandmark.LEFT_EAR)
    REAR = get_pt(mp_pose.PoseLandmark.RIGHT_EAR)
    LEYE = get_pt(mp_pose.PoseLandmark.LEFT_EYE)
    REYE = get_pt(mp_pose.PoseLandmark.RIGHT_EYE)

    # --------------------------------------------------------
    # Feature container (NaN by default)
    # --------------------------------------------------------
    res = {
        "headForwardDepth": np.nan,
        "headHeight": np.nan,
        "shoulderTilt": np.nan,
        "torsoRotation": np.nan,
        "shoulderWidth": np.nan,

        "theta_shoulders": np.nan,
        "thetaNeck": np.nan,
        "thetaNeck_rel": np.nan,

        "theta_RS_proj": np.nan,
        "theta_LS_proj": np.nan,
        "theta_RS_proj_relShoulders": np.nan,
        "theta_LS_proj_relShoulders": np.nan,

        "theta_ears": np.nan,
        "theta_ears_rel": np.nan,

        "theta_LEar_LS": np.nan,
        "theta_LEar_LS_relShoulders": np.nan,
        "theta_REar_RS": np.nan,
        "theta_REar_RS_relShoulders": np.nan,
    }

    # --------------------------------------------------------
    # Raw landmark storage
    # --------------------------------------------------------
    add_landmark_fields(res, "nose", NOSE)
    add_landmark_fields(res, "leftEar", LEAR)
    add_landmark_fields(res, "rightEar", REAR)
    add_landmark_fields(res, "leftShoulder", LS)
    add_landmark_fields(res, "rightShoulder", RS)
    add_landmark_fields(res, "leftEye", LEYE)
    add_landmark_fields(res, "rightEye", REYE)

    # --------------------------------------------------------
    # Geometry (requires shoulders)
    # --------------------------------------------------------

    theta_ears = None
    theta_L = None
    theta_R = None

    # Ear line (roll)
    if LEAR and REAR:
        theta_ears = np.arctan2(REAR.y - LEAR.y, REAR.x - LEAR.x)
        res["theta_ears"] = np.degrees(theta_ears)

    # Per-side ear–shoulder angles
    if LEAR:
        theta_L = np.arctan2(LEAR.x - LS.x, LEAR.y - LS.y)
        res["theta_LEar_LS"] = np.degrees(theta_L)

    if REAR:
        theta_R = np.arctan2(REAR.x - RS.x, REAR.y - RS.y)
        res["theta_REar_RS"] = np.degrees(theta_R)

    if LS and RS:
        shoulder_width = np.hypot(RS.x - LS.x, RS.y - LS.y)

        if shoulder_width > 0:
            Sx = (LS.x + RS.x) / 2
            Sy = (LS.y + RS.y) / 2
            Sz = (LS.z + RS.z) / 2

            res["shoulderWidth"] = shoulder_width
            res["shoulderTilt"] = (LS.y - RS.y) / shoulder_width
            res["torsoRotation"] = RS.z - LS.z

            theta_shoulders = np.arctan2(RS.y - LS.y, RS.x - LS.x)
            res["theta_shoulders"] = np.degrees(theta_shoulders)

            if LEAR and REAR:
                # theta_ears_rel just if the shoulders appear
                res["theta_ears_rel"] = np.degrees(
                    wrap_to_pi(theta_ears - theta_shoulders)
                )

            if LEAR:
                # theta_LEar_LS_relShoulders just if the shoulders appear
                res["theta_LEar_LS_relShoulders"] = np.degrees(
                    wrap_to_pi(theta_L - theta_shoulders)
                )

            if REAR:
                # theta_REar_RS_relShoulders just if the shoulders appear
                res["theta_REar_RS_relShoulders"] = np.degrees(
                    wrap_to_pi(theta_R - theta_shoulders)
                )

            # Shoulder projections (XY)
            theta_RS_proj = np.arctan2(RS.x - Sx, RS.y - Sy)
            theta_LS_proj = np.arctan2(LS.x - Sx, LS.y - Sy)

            res["theta_RS_proj"] = np.degrees(theta_RS_proj)
            res["theta_LS_proj"] = np.degrees(theta_LS_proj)

            res["theta_RS_proj_relShoulders"] = np.degrees(
                wrap_to_pi(theta_RS_proj - theta_shoulders)
            )
            res["theta_LS_proj_relShoulders"] = np.degrees(
                wrap_to_pi(theta_LS_proj - theta_shoulders)
            )

            # Neck features
            if NOSE:
                theta_neck = np.arctan2(NOSE.x - Sx, NOSE.y - Sy)
                res["thetaNeck"] = np.degrees(theta_neck)
                res["thetaNeck_rel"] = np.degrees(
                    wrap_to_pi(theta_neck - theta_shoulders)
                )

                res["headForwardDepth"] = (Sz - NOSE.z) / shoulder_width
                res["headHeight"] = (Sy - NOSE.y) / shoulder_width



    return res

# =========================
# CVAT label lookup
# =========================
def get_label_lookup(json_path):
    if not os.path.exists(json_path):
        print(f"❌ JSON not found: {json_path}")
        return {}

    try:
        with open(json_path, 'r', encoding="utf-8") as f:
            data = json.load(f)

        # 1. Map Category IDs to Names (e.g., 1 -> "Good_Posture")
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}

        # 2. Map Image IDs to Filenames (stripping extension)
        images = {img['id']: os.path.splitext(img['file_name'])[0] for img in data.get('images', [])}

        # 3. Build the lookup dictionary
        lookup = {}
        for ann in data.get('annotations', []):
            img_id = ann.get('image_id')
            cat_id = ann.get('category_id')

            filename_base = images.get(img_id)
            label_name = categories.get(cat_id, "Unlabeled")

            if filename_base:
                lookup[filename_base] = label_name

        if not lookup:
            print("⚠️ Warning: JSON loaded but 0 annotations were found in the 'annotations' list.")
        else:
            print(f"✅ Successfully matched {len(lookup)} labels from JSON.")

        return lookup
    except Exception as e:
        print(f"❌ Failed to parse COCO JSON: {e}")
        return {}# =========================
# Draw features on image
# =========================
def draw_feature_text(img_bgr, label, features_dict):
    img = img_bgr.copy()
    x0, y0 = 10, 25
    line_h = 20

    def fmt(v):
        if v is None:
            return "None"
        try:
            return f"{float(v):+.3f}"
        except Exception:
            return str(v)

    lines = [
        f"Label: {label}",

        # Basic normalized geometry
        f"headForwardDepth:     {fmt(features_dict.get('headForwardDepth'))}",
        f"headHeight:           {fmt(features_dict.get('headHeight'))}",
        f"shoulderWidth:        {fmt(features_dict.get('shoulderWidth'))}",
        f"shoulderTilt:         {fmt(features_dict.get('shoulderTilt'))}",
        f"torsoRotation:        {fmt(features_dict.get('torsoRotation'))}",

        # Main global angles
        f"theta_shoulders(deg): {fmt(features_dict.get('theta_shoulders'))}",
        f"thetaNeck(deg):       {fmt(features_dict.get('thetaNeck'))}",
        f"thetaNeck_rel(deg):   {fmt(features_dict.get('thetaNeck_rel'))}",

        # Ears & shoulders (roll / lateral)
        f"theta_ears(deg):      {fmt(features_dict.get('theta_ears'))}",
        f"theta_ears_rel(deg):  {fmt(features_dict.get('theta_ears_rel'))}",

        f"theta_LEar_LS(deg):   {fmt(features_dict.get('theta_LEar_LS'))}",
        f"theta_LEar_LS_rel(deg): {fmt(features_dict.get('theta_LEar_LS_relShoulders'))}",
        f"theta_REar_RS(deg):   {fmt(features_dict.get('theta_REar_RS'))}",
        f"theta_REar_RS_rel(deg): {fmt(features_dict.get('theta_REar_RS_relShoulders'))}",

        # Shoulder projections
        f"theta_RS_proj(deg):   {fmt(features_dict.get('theta_RS_proj'))}",
        f"theta_LS_proj(deg):   {fmt(features_dict.get('theta_LS_proj'))}",
        f"RS_proj_rel(deg):     {fmt(features_dict.get('theta_RS_proj_relShoulders'))}",
        f"LS_proj_rel(deg):     {fmt(features_dict.get('theta_LS_proj_relShoulders'))}",
    ]

    box_h = line_h * len(lines) + 10
    # Slightly wider box to fit longer lines
    cv2.rectangle(img, (5, 5), (650, 5 + box_h), (0, 0, 0), thickness=-1)

    for i, s in enumerate(lines):
        cv2.putText(
            img,
            s,
            (x0, y0 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    return img


# =========================
# Natural Sorting Helper
# =========================
def natural_sort_key(s):
    """
    Function to create a key for natural sorting.
    Treats 'frame_2' as smaller than 'frame_10'.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


# =========================
# Preview 10 images
# =========================
def preview_10_images_with_features(img_dir, json_path, preview_dir, n=10, pick="random"):
    os.makedirs(preview_dir, exist_ok=True)
    labels = get_label_lookup(json_path)

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"❌ No images found in: {img_dir}")
        return

    # SORT FILES NATURALLY
    files.sort(key=natural_sort_key)

    chosen = random.sample(files, k=min(n, len(files))) if pick == "random" else files[:min(n, len(files))]

    # ... (rest of preview function logic stays the same)


# =========================
# Full dataset generation + check images
# =========================
def run_integration(img_dir, json_path, output_file, output_landmark_dir):
    if not os.path.exists(img_dir):
        print(f"❌ Image dir not found: {img_dir}")
        return

    os.makedirs(output_landmark_dir, exist_ok=True)

    labels = get_label_lookup(json_path)
    dataset = []

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # NATURAL SORTING
    files.sort(key=natural_sort_key)

    print(f"📌 Found {len(files)} images in {img_dir}")

    for filename in files:
        img_id = os.path.splitext(filename)[0]
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        # Process the image with MediaPipe
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Initialize row with ID and Label
        row = {"Image_ID": filename, "Label": labels.get(img_id, "Missing_Label")}

        annotated = img.copy()

        if res.pose_landmarks:
            # Extract the actual math features
            features = extract_math_features(res.pose_landmarks.landmark)
            row.update(features)

            # Draw landmarks for the "check" folder
            mp_drawing.draw_landmarks(
                annotated,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            cv2.putText(annotated, "NO POSE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Save the landmark check image
        cv2.imwrite(os.path.join(output_landmark_dir, f"check_{filename}"), annotated)

        # Add the data to our list
        dataset.append(row)

    # Save to Excel with a safety check for open files
    try:
        pd.DataFrame(dataset).to_excel(output_file, index=False)
        print(f"✅ Done: {output_file} (rows={len(dataset)})")
    except PermissionError:
        print(f"❌ PERMISSION ERROR: Close '{output_file}' in Excel and run again!")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    main_output_file = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
                        r"\All individual Datasets - excel sheets")
    main_output_landmark_dir = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
                                r"\All individual Landmark Detected photos - from each video")
    main_preview_dir = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
                        r"\All Preview Features - for each video")

    # Sophie Fradkin video - Dataset:
    name = "Sophie Fradkin video"
    image_dir = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
                 r"\extract_frames_from_video-All_output_frames\Output_sophie_fradkin_video")
    json_path = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
                 r"\All jason file labeling - for all videos\test_jason\labelling_output_sophie_fradkin_video.json")

    output_file = Path(main_output_file) / f"Output_{name}_Dataset.xlsx"  # Folder for extracted frames
    output_landmark_dir = Path(main_output_landmark_dir) / f"Output_{name}_Landmark_Detected_Photos"
    preview_dir = Path(main_preview_dir) / f"Output_{name}_Preview_Features"

    # Shirel's first video file - Dataset:
    # name = "Shirel_labelling_for_cvat"
    # image_dir = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
    #              r"\extract_frames_from_video-All_output_frames\Output_video_for_cvat")
    # json_path = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
    #              r"\All jason file labeling - for all videos\shirel_labeling_for_cvat.json")
    #
    # output_file = Path(main_output_file)/f"Output_{name}_Dataset.xlsx"# Folder for extracted frames
    # output_landmark_dir = Path(main_output_landmark_dir)/f"Output_{name}_Landmark_Detected_Photos"
    # preview_dir = Path(main_preview_dir)/f"Output_{name}_Preview_Features"

    # Shirel's first video file - Dataset:
    # name = "Shirel1_good_and_bad_posture"
    # image_dir = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
    #              r"\extract_frames_from_video-All_output_frames\Output_implementing_bad_and_good_posture")
    # json_path = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames"
    #              r"\pythonProject1\All jason file labeling - for all videos\shirel_implementing_good_and_bad_posture1.json")
    #
    # output_file = Path(main_output_file)/f"Output_{name}_Dataset.xlsx"# Folder for extracted frames
    # output_landmark_dir = Path(main_output_landmark_dir)/f"Output_{name}_Landmark_Detected_Photos"
    # preview_dir = Path(main_preview_dir)/f"Output_{name}_Preview_Features"

    #Shirel's second video file - Dataset:
    # name = "Shirel2_good_posture"
    # image_dir = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames"
    #              r"\pythonProject1\extract_frames_from_video-All_output_frames\Output_implementing_good_posture")
    # json_path = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
    #              r"\All jason file labeling - for all videos\shirel_implementing_good_posture.json")
    #
    # output_file = Path(main_output_file)/f"Output_{name}_Dataset.xlsx"# Folder for extracted frames
    # output_landmark_dir = Path(main_output_landmark_dir)/f"Output_{name}_Landmark_Detected_Photos"
    # preview_dir = Path(main_preview_dir)/f"Output_{name}_Preview_Features"

    # Ester_Fradkin_labeling_first_video - Dataset:
    # name = "Ester_Fradkin_labeling_first_video"
    # image_dir = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
    #              r"\extract_frames_from_video-All_output_frames\Output_Video_Ester")
    # json_path = (r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1"
    #              r"\All jason file labeling - for all videos\Ester_Fradkin_labeling_first_video.json")
    #
    # output_file = Path(main_output_file) / f"Output_{name}_Dataset.xlsx"  # Folder for extracted frames
    # output_landmark_dir = Path(main_output_landmark_dir) / f"Output_{name}_Landmark_Detected_Photos"
    # preview_dir = Path(main_preview_dir) / f"Output_{name}_Preview_Features"

    runs = [
        {
            "NAME": name,
            "IMAGE_DIR": image_dir,
            "JSON_PATH": json_path,
            "OUTPUT_FILE": output_file,
            "OUTPUT_LANDMARK_DIR": output_landmark_dir,
            "PREVIEW_DIR": preview_dir
        }
        # {
        #     "NAME": "Shirel_1",
        #     "IMAGE_DIR": r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\Output_shirel_video1",
        #     "JSON_PATH": r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\shirel_implementing_good_and_bad_posture1.json",
        #     "OUTPUT_FILE": r"Shirel_1_Final_Posture_Dataset.xlsx",
        #     "OUTPUT_LANDMARK_DIR": r"Shirel_1_Landmark_Detected_Photos",
        #     "PREVIEW_DIR": r"Preview_Shirel1_Features"
        # }
    ]
    # runs = []
    for i, r in enumerate(runs, start=1):
        name = r.get("NAME", f"RUN_{i}")
        preview_dir = r.get("PREVIEW_DIR", f"Preview_{name}")

        print(f"\n================ RUN {i}/{len(runs)} : {name} ================")

        run_integration(
            r["IMAGE_DIR"],
            r["JSON_PATH"],
            r["OUTPUT_FILE"],
            r["OUTPUT_LANDMARK_DIR"]
        )

        preview_10_images_with_features(
            r["IMAGE_DIR"],
            r["JSON_PATH"],
            preview_dir,
            n=10,
            pick="random"  # or "first"
        )
