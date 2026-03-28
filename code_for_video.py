import cv2
import mediapipe as mp
import os

# =========================
# MediaPipe setup (from STEP 3)
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Same config as your STEP 3 script
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.01,
    min_tracking_confidence=0.01
)

# =========================
# Process many images
# =========================
def process_frames_folder(input_dir, output_dir):
    """
    Read all images from input_dir, run MediaPipe Pose with the SAME
    configuration as STEP 3, and save images with pose landmarks drawn
    into output_dir.
    """
    if not os.path.exists(input_dir):
        print(f"❌ Input folder does not exist: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Allowed image extensions
    exts = (".png", ".jpg", ".jpeg", ".bmp")

    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(exts)]

    if not files:
        print(f"❌ No image files found in: {input_dir}")
        return

    print(f"📌 Found {len(files)} images in {input_dir}")

    for filename in files:
        in_path = os.path.join(input_dir, filename)
        img_bgr = cv2.imread(in_path)

        if img_bgr is None:
            print(f"⚠️ Could not read image: {in_path}")
            continue

        # Run MediaPipe Pose (same as STEP 3)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        annotated = img_bgr.copy()

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            cv2.putText(
                annotated,
                "NO POSE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        # Save with the same name into output_dir
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, annotated)
        print(f"✅ Saved: {out_path}")

    print("🎉 Done processing all images!")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # 🔴 CHANGE THESE PATHS
    INPUT_FRAMES_DIR = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\Output_video_for_cvat"
    OUTPUT_FRAMES_DIR = r"C:\Users\ester\Desktop\Ai posture fixer - code project\video_to_frames\pythonProject1\Cvat_for_video_shirel_output_with_landmarks"

    process_frames_folder(INPUT_FRAMES_DIR, OUTPUT_FRAMES_DIR)
