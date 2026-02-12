import cv2
import numpy as np


def _normalize_illumination(image_bgr, face_box):
    """Normalize illumination across the face to even out shadow/highlight asymmetry.

    Uses homomorphic-style filtering: divides out the slowly-varying illumination
    field (estimated via heavy Gaussian blur) and replaces it with the face mean.
    This evens out left/right lighting differences without changing overall brightness.
    """
    if face_box is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    fx, fy, fw, fh = face_box

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)

    # Heavy blur to estimate the illumination field
    # Kernel proportional to face size so it captures lighting gradients
    ksize = max(fw, fh) | 1  # ensure odd
    if ksize < 3:
        return image_bgr
    illumination = cv2.GaussianBlur(l, (ksize, ksize), ksize / 3)
    illumination = np.maximum(illumination, 1.0)

    # Target brightness: mean of the face region (preserves skin tone)
    x1, y1 = max(0, fx), max(0, fy)
    x2, y2 = min(w, fx + fw), min(h, fy + fh)
    target = np.mean(l[y1:y2, x1:x2])

    # Normalize: (original / illumination_field) * target_mean
    normalized = (l / illumination) * target

    # Blend with Gaussian falloff from face center — strong near face, none far away
    face_cx, face_cy = fx + fw // 2, fy + fh // 2
    sigma_x, sigma_y = fw * 1.2, fh * 1.2
    y_coords, x_coords = np.ogrid[:h, :w]
    blend_mask = np.exp(
        -((x_coords - face_cx) ** 2 / (2 * sigma_x**2)
          + (y_coords - face_cy) ** 2 / (2 * sigma_y**2))
    ).astype(np.float32)

    # 65% correction at face center, fading to 0% away from face
    strength = 0.65
    mask = blend_mask * strength
    l_result = l * (1 - mask) + normalized * mask

    lab[:, :, 0] = np.clip(l_result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _apply_clahe(image_bgr, face_box):
    """Gentle CLAHE on LAB L-channel for micro-contrast consistency."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # Blend at 30% — just enough to lift deep shadows slightly
    l_result = cv2.addWeighted(l_channel, 0.7, l_enhanced, 0.3, 0)

    lab[:, :, 0] = l_result
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _white_balance(image_bgr, face_box, strength=0.15):
    """Gray-world white balance using face ROI, blended gently."""
    if face_box is None:
        roi = image_bgr
    else:
        fx, fy, fw, fh = face_box
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, fx), max(0, fy)
        x2, y2 = min(w, fx + fw), min(h, fy + fh)
        roi = image_bgr[y1:y2, x1:x2]

    avg_b, avg_g, avg_r = cv2.mean(roi)[:3]
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        return image_bgr

    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    result = image_bgr.astype(np.float32)
    balanced = result.copy()
    balanced[:, :, 0] *= scale_b
    balanced[:, :, 1] *= scale_g
    balanced[:, :, 2] *= scale_r
    balanced = np.clip(balanced, 0, 255)

    blended = result * (1 - strength) + balanced * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def normalize_lighting(image_bgr, face_box=None):
    """Normalize lighting: even out shadows, gentle contrast, slight color correction.

    Pipeline:
      1. Illumination normalization — evens left/right shadow asymmetry
      2. Gentle CLAHE — lifts micro-contrast in shadows
      3. Light white balance — corrects color cast
    """
    result = _normalize_illumination(image_bgr, face_box)
    result = _apply_clahe(result, face_box)
    result = _white_balance(result, face_box, strength=0.15)
    return result
