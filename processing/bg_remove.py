import cv2
import numpy as np
from rembg import new_session, remove


_session = None


def init_session():
    """Initialize rembg GPU session with BiRefNet portrait model.

    Call once at startup to pre-download model and warm up GPU.
    """
    global _session
    if _session is None:
        _session = new_session(
            model_name="birefnet-portrait",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    return _session


def remove_background(image_bgr):
    """Remove background from BGR image.

    Returns RGBA numpy array with transparent background.
    """
    session = init_session()

    # rembg expects BGR input when given numpy array
    rgba = remove(
        image_bgr,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        post_process_mask=True,
    )
    return rgba


def composite_on_white(rgba_image):
    """Alpha-composite RGBA image onto white background.

    Returns BGR numpy array.
    """
    if rgba_image.shape[2] == 3:
        return rgba_image

    # Split channels
    rgb = rgba_image[:, :, :3].astype(np.float32)
    alpha = rgba_image[:, :, 3:4].astype(np.float32) / 255.0

    # Composite onto white
    white = np.full_like(rgb, 255.0)
    composited = rgb * alpha + white * (1.0 - alpha)

    return composited.astype(np.uint8)
