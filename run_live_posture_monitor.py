# STEP 5 – LIVE POSTURE INFERENCE
#
# Input:
#   - Live webcam video stream
#   - Saved posture-classification black box (joblib)
#
# Output:
#   - Real-time posture classification (Good / Bad)
#   - Smoothed stable posture state
#   - Color-coded visual feedback (Green = Good, Red = Bad)
#   - Timer showing seconds since last state change
#
# Description:
#   This script extracts pose features from each webcam frame using MediaPipe,
#   classifies posture using a trained KNN model, applies asymmetric smoothing
#   (slow Bad detection, fast Good recovery), and displays live feedback.

import cv2
import joblib
import numpy as np
from collections import deque
import time
import mediapipe as mp

# =========================
# MediaPipe Pose (LIVE)
# =========================
mp_pose = mp.solutions.pose

pose_live = mp_pose.Pose(
    static_image_mode=False,      # enable tracking
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# 1) LOAD BLACK BOX MODEL
# =========================
PACK_PATH = "posture_knn_blackbox.joblib"
pack = joblib.load(PACK_PATH)

imputer = pack["imputer"]   # <<< ADD THIS
scaler = pack["scaler"]
model = pack["model"]
feature_names = pack["feature_names"]
classes = pack.get("classes", None)

print("Loaded model:", PACK_PATH)
print("Num features:", len(feature_names))
print("Classes:", classes)

# =========================
# 2) STATE & DECISION LOGIC
# =========================
FPS = 30
WINDOW_SEC = 2.0
WINDOW_SIZE = int(FPS * WINDOW_SEC)

# WINDOW_SIZE = 60          # ~2 seconds at 30 FPS (slow Bad detection)
BAD_ON_RATIO = 0.7         # Good → Bad threshold
GOOD_RECOVERY_FRAMES = 20  # ~0.6 sec Good streak for recovery

pred_window = deque(maxlen=WINDOW_SIZE)

stable_state = "Good"
stable_state_since = time.time()
good_streak = 0

def normalize_label(lbl):
    """Normalize model output into 'Good' / 'Bad'."""
    s = str(lbl).strip().lower()
    if "bad" in s or s == "0":
        return "Bad"
    if "good" in s or s == "1":
        return "Good"
    return "Unknown"

def state_color(state):
    """Color for UI based on stable state (BGR)."""
    if state == "Good":
        return (0, 255, 0)    # Green
    if state == "Bad":
        return (0, 0, 255)    # Red
    return (255, 255, 255)   # White fallback

def update_stable_state(frame_label):
    """
    Update stable posture state using:
    - Long sliding window for Good → Bad
    - Short Good streak for Bad → Good
    """
    global stable_state, stable_state_since, good_streak

    lbl = normalize_label(frame_label)
    pred_window.append(lbl)

    bad_count = sum(1 for x in pred_window if x == "Bad")
    bad_ratio = bad_count / len(pred_window)

    # --- GOOD → BAD (slow, conservative) ---
    if bad_ratio >= BAD_ON_RATIO and stable_state != "Bad":
        stable_state = "Bad"
        stable_state_since = time.time()
        good_streak = 0
        return stable_state, bad_ratio

    # --- BAD → GOOD (fast recovery) ---
    if stable_state == "Bad":
        if lbl == "Good":
            good_streak += 1
        else:
            good_streak = 0

        if good_streak >= GOOD_RECOVERY_FRAMES:
            stable_state = "Good"
            stable_state_since = time.time()
            good_streak = 0

    return stable_state, bad_ratio

def too_many_missing(x, threshold=0.4):
    return np.mean(np.isnan(x)) > threshold


# =========================
# 3) BLACK BOX PREDICTION
# =========================
def predict_from_features_dict(features_dict):
    """
    features_dict -> ordered vector -> scaler -> model.predict
    Returns: (label, prob_bad or None)
    """
    x = np.array(
        [features_dict[name] for name in feature_names],
        dtype=float
    ).reshape(1, -1)

    if too_many_missing(x):
        return "Unknown", None

    # >>> ADD THESE TWO LINES <<<
    x_imputed = imputer.transform(x)
    x_scaled = scaler.transform(x_imputed)

    # x_scaled = scaler.transform(x)
    label = model.predict(x_scaled)[0]

    prob_bad = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_scaled)[0]
        cls = list(model.classes_)
        if "Bad" in cls:
            prob_bad = float(probs[cls.index("Bad")])
        elif "bad" in cls:
            prob_bad = float(probs[cls.index("bad")])

    return label, prob_bad

# =========================
# 4) FEATURE EXTRACTION
# =========================

# >>> ADDED
def wrap_to_pi(angle):
    """Wrap angle (radians) to range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def extract_features_from_frame(frame):
    """
    Extract posture features from a single frame.
    Output keys exactly match feature_names used in training.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_live.process(rgb)

    if not result.pose_landmarks:
        return {name: np.nan for name in feature_names}

    landmarks = result.pose_landmarks.landmark

    def get_pt(idx):
        pt = landmarks[idx.value]
        return pt if pt.visibility > 0.1 else None

    def add_landmark(res, name, lm):
        res[f"{name}_x"] = np.nan if lm is None else float(lm.x)
        res[f"{name}_y"] = np.nan if lm is None else float(lm.y)
        res[f"{name}_z"] = np.nan if lm is None else float(lm.z)
        res[f"{name}_vis"] = np.nan if lm is None else float(lm.visibility)

    LS = get_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    RS = get_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    NOSE = get_pt(mp_pose.PoseLandmark.NOSE)
    LEAR = get_pt(mp_pose.PoseLandmark.LEFT_EAR)
    REAR = get_pt(mp_pose.PoseLandmark.RIGHT_EAR)
    LEYE = get_pt(mp_pose.PoseLandmark.LEFT_EYE)
    REYE = get_pt(mp_pose.PoseLandmark.RIGHT_EYE)

    res = {
        "headForwardDepth": np.nan,
        "shoulderTilt": np.nan,
        "thetaNeck": np.nan,
        "thetaNeck_rel": np.nan, # >>> ADDED
        "headHeight": np.nan,
        "torsoRotation": np.nan,
        "theta_shoulders": np.nan, # >>> ADDED
        "theta_RS_proj": np.nan, # >>> ADDED
        "theta_LS_proj": np.nan, # >>> ADDED
        "theta_RS_proj_relShoulders": np.nan, # >>> ADDED
        "theta_LS_proj_relShoulders": np.nan, # >>> ADDED
        "theta_ears": np.nan, # >>> ADDED
        "theta_ears_rel": np.nan, # >>> ADDED
        "theta_LEar_LS": np.nan, # >>> ADDED
        "theta_REar_RS": np.nan, # >>> ADDED
        "theta_LEar_LS_relShoulders": np.nan, # >>> ADDED
        "theta_REar_RS_relShoulders": np.nan # >>> ADDED
    }

    add_landmark(res, "nose", NOSE)
    add_landmark(res, "leftEar", LEAR)
    add_landmark(res, "rightEar", REAR)
    add_landmark(res, "leftShoulder", LS)
    add_landmark(res, "rightShoulder", RS)
    add_landmark(res, "leftEye", LEYE)
    add_landmark(res, "rightEye", REYE)

    if LS and RS:
        shoulder_width = np.hypot(RS.x - LS.x, RS.y - LS.y)

        if shoulder_width > 0:
            Sx = (LS.x + RS.x) / 2
            Sy = (LS.y + RS.y) / 2
            Sz = (LS.z + RS.z) / 2

            res["shoulderTilt"] = (LS.y - RS.y) / shoulder_width
            res["torsoRotation"] = RS.z - LS.z

            # >>> ADDED
            theta_shoulders = np.arctan2(RS.y - LS.y, RS.x - LS.x)
            res["theta_shoulders"] = float(np.degrees(theta_shoulders))

            # >>> ADDED: raw shoulder projection angles (XY)
            theta_RS_proj = np.arctan2(RS.x - Sx, RS.y - Sy)
            theta_LS_proj = np.arctan2(LS.x - Sx, LS.y - Sy)

            res["theta_RS_proj"] = float(np.degrees(theta_RS_proj))
            res["theta_LS_proj"] = float(np.degrees(theta_LS_proj))

            # >>> ADDED: normalized by shoulder line
            res["theta_RS_proj_relShoulders"] = float(
                np.degrees(wrap_to_pi(theta_RS_proj - theta_shoulders))
            )
            res["theta_LS_proj_relShoulders"] = float(
                np.degrees(wrap_to_pi(theta_LS_proj - theta_shoulders))
            )

            if NOSE:
                theta_neck = np.arctan2(NOSE.x - Sx, NOSE.y - Sy)
                res["thetaNeck"] = float(np.degrees(theta_neck))

                # >>> ADDED: neck angle relative to shoulders
                theta_neck_rel = wrap_to_pi(theta_neck - theta_shoulders)
                res["thetaNeck_rel"] = float(np.degrees(theta_neck_rel))

                res["headForwardDepth"] = (Sz - NOSE.z) / shoulder_width
                res["headHeight"] = (Sy - NOSE.y) / shoulder_width

            # >>> ADDED: head roll from ear line (XY)
            if LEAR is not None and REAR is not None:
                theta_ears = np.arctan2(REAR.y - LEAR.y, REAR.x - LEAR.x)
                res["theta_ears"] = float(np.degrees(theta_ears))
                res["theta_ears_rel"] = float(
                    np.degrees(wrap_to_pi(theta_ears - theta_shoulders))
                )

            # >>> ADDED: per-side ear–shoulder angles
            if LEAR is not None:
                theta_LEar_LS = np.arctan2(LEAR.x - LS.x, LEAR.y - LS.y)
                res["theta_LEar_LS"] = float(np.degrees(theta_LEar_LS))
                res["theta_LEar_LS_relShoulders"] = float(
                    np.degrees(wrap_to_pi(theta_LEar_LS - theta_shoulders))
                )

            if REAR is not None:
                theta_REar_RS = np.arctan2(REAR.x - RS.x, REAR.y - RS.y)
                res["theta_REar_RS"] = float(np.degrees(theta_REar_RS))
                res["theta_REar_RS_relShoulders"] = float(
                    np.degrees(wrap_to_pi(theta_REar_RS - theta_shoulders))
                )

    return {name: res.get(name, np.nan) for name in feature_names}

# =========================
# 5) MAIN LIVE LOOP
# =========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        features = extract_features_from_frame(frame)

        if features is not None:
            frame_label, prob_bad = predict_from_features_dict(features)
            stable, bad_ratio = update_stable_state(frame_label)

            elapsed_sec = time.time() - stable_state_since
            color = state_color(stable)

            cv2.putText(frame, f"Frame: {frame_label}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame,
                        f"Stable: {stable}  bad_ratio={bad_ratio:.2f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

            cv2.putText(frame,
                        f"Time in state: {elapsed_sec:.1f}s",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if prob_bad is not None:
                cv2.putText(frame,
                            f"ProbBad: {prob_bad:.2f}",
                            (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(frame,
                        "No pose detected",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Live Posture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()