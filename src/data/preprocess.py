import cv2

def resize_keep_aspect(frame, target_long=640):
    h, w = frame.shape[:2]

    scale = target_long / max(h, w)

    new_w = round(w * scale)
    new_h = round(h * scale)

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    return resized, scale