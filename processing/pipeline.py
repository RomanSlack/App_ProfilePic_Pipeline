import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from .face_detect import detect_face, compute_headshot_crop, crop_headshot
from .bg_remove import init_session, remove_background
from .lighting import normalize_lighting

OUTPUT_SIZE = 800


def _circle_mask(size):
    """Create an antialiased circular alpha mask."""
    mask = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = size // 2
    cv2.circle(mask, (center, center), radius, 255, -1, lineType=cv2.LINE_AA)
    return mask


def init_models():
    """Pre-initialize all models. Call once at startup."""
    init_session()


def process_image(image_bytes):
    """Full pipeline: detect → crop → light → remove bg → circle mask → resize.

    Returns RGBA PNG with transparent background in a circular crop.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        dict with 'image' (base64 PNG) and 'metadata'.

    Raises:
        ValueError: If no face is detected.
    """
    # 1. Decode image + EXIF auto-orient
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 2. Detect face
    face_info = detect_face(image_bgr)
    if face_info is None:
        raise ValueError("No face detected in the image. Please try another photo.")

    # 3. Compute headshot crop + crop
    crop_coords = compute_headshot_crop(face_info, image_bgr.shape)
    cropped = crop_headshot(image_bgr, crop_coords)

    # Re-detect face in cropped image for accurate face box
    cropped_face = detect_face(cropped)
    face_box = cropped_face["box"] if cropped_face else None

    # 4. Normalize lighting (before bg removal — works better with full context)
    lit = normalize_lighting(cropped, face_box)

    # 5. Remove background → RGBA with transparency
    rgba = remove_background(lit)

    # 6. Resize to output size (RGBA)
    rgba = cv2.resize(rgba, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # 7. Apply circular mask to alpha channel
    circle = _circle_mask(OUTPUT_SIZE)
    if rgba.shape[2] == 4:
        rgba[:, :, 3] = cv2.bitwise_and(rgba[:, :, 3], circle)
    else:
        # If no alpha, create BGRA with circle as alpha
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = circle

    # 8. Encode to PNG (with alpha) and return base64
    _, png_bytes = cv2.imencode(".png", rgba)
    b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

    return {
        "image": b64,
        "metadata": {
            "original_size": f"{image_bgr.shape[1]}x{image_bgr.shape[0]}",
            "output_size": f"{OUTPUT_SIZE}x{OUTPUT_SIZE}",
        },
    }
