import os

import cv2
import numpy as np
import mediapipe as mp

_landmarker = None
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        base_options = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def detect_face(image_bgr):
    """Detect face and return bounding box + key landmarks.

    Returns dict with keys: box (x, y, w, h), landmarks dict,
    or None if no face found.
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = _get_landmarker().detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]

    # Extract key landmarks (pixel coords)
    landmarks = {}
    key_indices = {
        "forehead": 10,
        "chin": 152,
        "left_cheek": 234,
        "right_cheek": 454,
        "nose_tip": 1,
        "left_eye": 33,
        "right_eye": 263,
    }
    for name, idx in key_indices.items():
        lm = face[idx]
        landmarks[name] = (int(lm.x * w), int(lm.y * h))

    # Compute face bounding box from all landmarks
    xs = [int(lm.x * w) for lm in face]
    ys = [int(lm.y * h) for lm in face]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return {
        "box": (x_min, y_min, x_max - x_min, y_max - y_min),
        "landmarks": landmarks,
    }


def compute_headshot_crop(face_info, image_shape):
    """Compute a square crop region for a headshot.

    Face sits in upper ~55% of the crop, shoulders below.
    Returns (x, y, size) for a square crop.
    """
    h, w = image_shape[:2]
    lm = face_info["landmarks"]

    # Face dimensions from landmarks
    face_left = lm["left_cheek"][0]
    face_right = lm["right_cheek"][0]
    face_top = lm["forehead"][1]
    face_bottom = lm["chin"][1]
    face_width = face_right - face_left
    face_height = face_bottom - face_top
    face_cx = (face_left + face_right) // 2
    face_cy = (face_top + face_bottom) // 2

    # Crop size: face_height * 3.2 — zoomed out enough for full head in circle
    crop_size = int(face_height * 3.2)
    crop_size = max(crop_size, int(face_width * 2.4))
    crop_size = min(crop_size, min(w, h))

    # Position face center at 45% from top — keeps full crown inside circle curve
    crop_y = face_cy - int(crop_size * 0.55)
    crop_x = face_cx - crop_size // 2

    # Clamp to image bounds
    crop_x = max(0, min(crop_x, w - crop_size))
    crop_y = max(0, min(crop_y, h - crop_size))

    # Final clamp on size if near edges
    crop_size = min(crop_size, w - crop_x, h - crop_y)

    return crop_x, crop_y, crop_size


def crop_headshot(image_bgr, crop_coords):
    """Crop the image to the headshot region."""
    x, y, size = crop_coords
    return image_bgr[y : y + size, x : x + size].copy()
