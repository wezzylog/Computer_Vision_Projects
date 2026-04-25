"""
Eye that tracks finger movement using OpenCV + MediaPipe
Run: pip install opencv-python mediapipe
Then: python eye_tracker.py
"""

import cv2
import mediapipe as mp
import numpy as np
import math

# ── MediaPipe hands setup ──────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ── Window / canvas size ───────────────────────────────────────────────────────
W, H = 800, 600

# ── Eye drawing parameters ─────────────────────────────────────────────────────
EYE_CENTER   = (W // 2, H // 2)
EYE_RX       = 160          # eye outer ellipse X-radius
EYE_RY       = 110          # eye outer ellipse Y-radius
IRIS_R       = 55           # iris radius
PUPIL_R      = 22           # pupil radius
PUPIL_MAX    = IRIS_R - PUPIL_R - 6   # max pupil travel inside iris

# Palette
BG_COLOR        = (18,  15,  25)
WHITE_COLOR     = (230, 225, 215)
IRIS_OUTER      = (40,  120, 200)
IRIS_INNER      = (20,   70, 160)
PUPIL_COLOR     = (8,    8,   8)
HIGHLIGHT_COLOR = (255, 255, 255)
VEIN_COLOR      = (180,  80,  80)
EYELID_COLOR    = (60,   45,  55)
LASH_COLOR      = (30,   20,  30)
GLOW_COLOR      = (80,  140, 220)


def draw_eye(frame, gaze_target_norm):
    """
    Draw a realistic-ish eye that looks toward gaze_target_norm (x,y in -1..1).
    """
    cx, cy = EYE_CENTER

    # ── Iris / pupil offset ───────────────────────────────────────────────────
    gx = int(gaze_target_norm[0] * PUPIL_MAX * 0.85)
    gy = int(gaze_target_norm[1] * PUPIL_MAX * 0.55)   # vertical range smaller
    iris_cx, iris_cy = cx + gx, cy + gy

    # ── Soft glow behind eye ──────────────────────────────────────────────────
    for r_off, alpha in [(90, 0.06), (60, 0.10), (35, 0.14)]:
        overlay = frame.copy()
        cv2.ellipse(overlay, (cx, cy),
                    (EYE_RX + r_off, EYE_RY + r_off),
                    0, 0, 360, GLOW_COLOR, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Sclera (white of the eye) ─────────────────────────────────────────────
    cv2.ellipse(frame, (cx, cy), (EYE_RX, EYE_RY), 0, 0, 360, WHITE_COLOR, -1)

    # ── Subtle veins ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    for _ in range(8):
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.integers(30, 80)
        sx = cx + int(math.cos(angle) * 40)
        sy = cy + int(math.sin(angle) * 25)
        ex = sx + int(math.cos(angle) * length)
        ey = sy + int(math.sin(angle) * length)
        cv2.line(frame, (sx, sy), (ex, ey), VEIN_COLOR, 1)

    # ── Iris ──────────────────────────────────────────────────────────────────
    cv2.circle(frame, (iris_cx, iris_cy), IRIS_R, IRIS_OUTER, -1)
    # radial gradient-like rings
    for r, col in [
        (IRIS_R - 5,  (50, 130, 210)),
        (IRIS_R - 15, IRIS_INNER),
        (IRIS_R - 28, (15, 55, 130)),
    ]:
        cv2.circle(frame, (iris_cx, iris_cy), r, col, -1)

    # iris texture lines
    for i in range(20):
        angle = (i / 20) * 2 * math.pi
        r1, r2 = PUPIL_R + 4, IRIS_R - 4
        p1 = (int(iris_cx + math.cos(angle) * r1),
              int(iris_cy + math.sin(angle) * r1))
        p2 = (int(iris_cx + math.cos(angle) * r2),
              int(iris_cy + math.sin(angle) * r2))
        cv2.line(frame, p1, p2, (30, 90, 170), 1)

    # ── Pupil ─────────────────────────────────────────────────────────────────
    cv2.circle(frame, (iris_cx, iris_cy), PUPIL_R, PUPIL_COLOR, -1)

    # ── Specular highlights ───────────────────────────────────────────────────
    hx = iris_cx - IRIS_R // 4
    hy = iris_cy - IRIS_R // 4
    cv2.circle(frame, (hx, hy), 9, HIGHLIGHT_COLOR, -1)
    cv2.circle(frame, (hx + 14, hy + 12), 4, HIGHLIGHT_COLOR, -1)

    # ── Eyelid clip mask (top & bottom) ───────────────────────────────────────
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (EYE_RX, EYE_RY), 0, 0, 360, 255, -1)
    frame[:] = cv2.bitwise_and(frame, frame, mask=mask)

    # ── Eyelids ───────────────────────────────────────────────────────────────
    # Top lid
    pts_top = np.array([
        [cx - EYE_RX - 10, cy],
        [cx - EYE_RX // 2, cy - EYE_RY - 20],
        [cx,               cy - EYE_RY - 30],
        [cx + EYE_RX // 2, cy - EYE_RY - 20],
        [cx + EYE_RX + 10, cy],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_top], EYELID_COLOR)
    cv2.polylines(frame, [pts_top], False, LASH_COLOR, 2, cv2.LINE_AA)

    # Bottom lid
    pts_bot = np.array([
        [cx - EYE_RX - 10, cy],
        [cx - EYE_RX // 2, cy + EYE_RY + 10],
        [cx,               cy + EYE_RY + 15],
        [cx + EYE_RX // 2, cy + EYE_RY + 10],
        [cx + EYE_RX + 10, cy],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts_bot], EYELID_COLOR)

    # ── Eyelashes (top) ───────────────────────────────────────────────────────
    for i in range(13):
        t = (i / 12) * math.pi          # 0..π maps left-to-right along top lid
        lx = int(cx + EYE_RX * math.cos(math.pi - t))
        ly = int(cy - EYE_RY * abs(math.sin(t)) - 8)
        # lash direction: outward + slightly upward
        lash_angle = math.pi / 2 + (t - math.pi / 2) * 0.4
        llen = 18 + int(8 * abs(math.sin(t)))
        ex2 = int(lx + math.cos(math.pi - t - 0.3) * llen)
        ey2 = int(ly - math.sin(abs(t - math.pi / 2) * 0.5 + 0.6) * llen)
        cv2.line(frame, (lx, ly), (ex2, ey2), LASH_COLOR, 2, cv2.LINE_AA)


def get_index_tip_norm(hand_landmarks, frame_w, frame_h):
    """Return (nx, ny) normalised to -1..1 from index finger tip."""
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    # tip.x/y are already 0-1 from MediaPipe; flip x because webcam mirrors
    nx = (tip.x - 0.5) * 2      # -1 (left) .. +1 (right)
    ny = (tip.y - 0.5) * 2      # -1 (top)  .. +1 (bottom)
    return nx, ny


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    gaze = [0.0, 0.0]      # smoothed gaze direction
    SMOOTH = 0.15           # lerp factor (lower = lazier eye)

    print("👁  Eye tracker running — point your index finger at the camera.")
    print("Press  Q  to quit.")

    while True:
        ok, cam_frame = cap.read()
        if not ok:
            break

        cam_frame = cv2.flip(cam_frame, 1)   # mirror so it feels natural
        rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ── Determine gaze target ─────────────────────────────────────────────
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            tx, ty = get_index_tip_norm(lm, 640, 480)
            # Draw a small dot on the webcam feed for reference
            tip_px = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dot_x = int(tip_px.x * 640)
            dot_y = int(tip_px.y * 480)
            cv2.circle(cam_frame, (dot_x, dot_y), 10, (0, 255, 100), -1)
        else:
            tx, ty = 0.0, 0.0   # look centre when no hand detected

        # Smooth gaze
        gaze[0] += (tx - gaze[0]) * SMOOTH
        gaze[1] += (ty - gaze[1]) * SMOOTH

        # ── Draw eye canvas ───────────────────────────────────────────────────
        canvas = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        draw_eye(canvas, gaze)

        # Small camera preview inset
        thumb = cv2.resize(cam_frame, (200, 150))
        canvas[H - 160:H - 10, W - 210:W - 10] = thumb
        cv2.rectangle(canvas, (W - 211, H - 161), (W - 9, H - 9),
                      (80, 80, 100), 1)
        cv2.putText(canvas, "camera", (W - 205, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 160), 1)

        # HUD
        cv2.putText(canvas, "Point your index finger at the camera",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 180), 1)
        cv2.putText(canvas, f"gaze  x={gaze[0]:+.2f}  y={gaze[1]:+.2f}",
                    (20, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (100, 100, 130), 1)

        cv2.imshow("Eye Tracker", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
