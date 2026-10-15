"""Invisibility cloak effect with OpenCV.

Controls (click the video window first):
    1 / 2 / 3  - switch cloak color (red / green / blue)
    b          - re-capture the background (step out of the frame!)
    q or ESC   - quit
"""
import time

import cv2
import numpy as np

# HSV ranges per cloak color. Red wraps around the hue axis, so it needs two ranges.
COLOR_RANGES = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "green": [((35, 80, 50), (85, 255, 255))],
    "blue": [((90, 80, 50), (130, 255, 255))],
}
KEY_TO_COLOR = {ord("1"): "red", ord("2"): "green", ord("3"): "blue"}

KERNEL = np.ones((3, 3), np.uint8)


def capture_background(cap, n_frames=30, warmup=15):
    """Return a median-averaged background (cleaner than a single frame)."""
    for _ in range(warmup):  # let auto-exposure / white balance settle
        cap.read()

    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        time.sleep(0.02)

    if not frames:
        raise RuntimeError("Could not capture any background frames.")

    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return cv2.flip(background, 1)


def build_cloak_mask(hsv, color):
    """Binary-ish mask (0-255, feathered edges) of the cloak-colored pixels."""
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lower, upper in COLOR_RANGES[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))

    # Remove speckle noise, fill small holes, then grow slightly to cover edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.dilate(mask, KERNEL, iterations=2)

    # Feather the edges so the blend doesn't look jagged
    return cv2.GaussianBlur(mask, (9, 9), 0)


def apply_cloak(frame, background, mask):
    """Alpha-blend background into the frame wherever the mask is set."""
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    blended = alpha * background + (1.0 - alpha) * frame
    return blended.astype(np.uint8)


def main(camera_index=0, output_path="output.avi"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}.")

    # Use the camera's real resolution/FPS so the output file is always valid
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
    )

    color = "red"
    print("Step out of the frame - capturing background...")
    background = capture_background(cap)
    print(__doc__)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror view
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = build_cloak_mask(hsv, color)
            result = apply_cloak(frame, background, mask)

            out.write(result)
            cv2.putText(result, f"Cloak: {color}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Invisibility Cloak", result)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("b"):
                background = capture_background(cap)
            elif key in KEY_TO_COLOR:
                color = KEY_TO_COLOR[key]
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
