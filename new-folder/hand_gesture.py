import ctypes
import sys

# --- Experimental workaround for Python 3.14 + MediaPipe on Windows ---
# MediaPipe's loader looks up a function called "free" in its own DLL, which
# doesn't export it on Windows. If the lookup fails, fall back to the
# Windows C runtime's free().
if sys.platform == "win32":
    _orig_getitem = ctypes.CDLL.__getitem__

    def _patched_getitem(self, name):
        try:
            return _orig_getitem(self, name)
        except AttributeError:
            if name == "free":
                return ctypes.CDLL("ucrtbase")["free"]
            raise

    ctypes.CDLL.__getitem__ = _patched_getitem
# -----------------------------------------------------------------------

import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def open_camera():
    """Try several backends and camera indexes until one works."""
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        for index in (0, 1, 2, 3):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"Using camera index {index} (backend {backend})")
                    return cap
            cap.release()
    return None


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def fingers_up(lm):
    wrist = lm[0]
    pinky_base = lm[17]
    thumb = distance(lm[4], pinky_base) > distance(lm[3], pinky_base) * 1.1
    others = [
        distance(lm[tip], wrist) > distance(lm[pip], wrist)
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
    ]
    return [thumb] + others


def classify(lm):
    pattern = fingers_up(lm)
    thumb, index, middle, ring, pinky = pattern
    palm_size = distance(lm[0], lm[9])
    pinch = distance(lm[4], lm[8]) / palm_size

    if pinch < 0.25 and middle and ring and pinky:
        return "OK"
    if pattern == [False, False, False, False, False]:
        return "Fist"
    if pattern == [True, True, True, True, True]:
        return "Open Palm"
    if pattern == [False, True, False, False, False]:
        return "Pointing"
    if pattern == [False, True, True, False, False]:
        return "Peace"
    if pattern == [False, True, True, True, False]:
        return "Three"
    if pattern == [False, True, True, True, True]:
        return "Four"
    if pattern == [False, True, False, False, True]:
        return "Rock"
    if pattern == [True, True, False, False, True]:
        return "I Love You"
    if pattern == [True, False, False, False, False]:
        return "Thumbs Up" if lm[4].y < lm[0].y else "Thumbs Down"
    return "Unknown"


def draw_hand(frame, lm):
    h, w, _ = frame.shape
    points = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (255, 255, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
    return points


def main():
    ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = open_camera()
    if cap is None:
        raise RuntimeError(
            "Could not open webcam. Close other apps using the camera and "
            "check Settings > Privacy & security > Camera."
        )

    prev_time = 0
    start = time.monotonic()

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for lm, handedness in zip(result.hand_landmarks, result.handedness):
                points = draw_hand(frame, lm)
                gesture = classify(lm)
                side = handedness[0].category_name

                x, y = points[0]
                label = f"{side}: {gesture}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                top_left = (max(x - 10, 0), max(y + 20, 0))
                bottom_right = (top_left[0] + tw + 20, top_left[1] + th + 16)
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    label,
                    (top_left[0] + 10, top_left[1] + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            now = time.time()
            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
