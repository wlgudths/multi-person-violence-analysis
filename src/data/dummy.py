import cv2
import numpy as np

def dummy_img():
    dummy = np.zeros(
        (640, 640, 3),
        dtype=np.uint8
    )

    cv2.circle(
        dummy,
        (320, 180),
        40,
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        dummy,
        (280, 220),
        (360, 430),
        (255, 255, 255),
        -1
    )

    cv2.line(
        dummy,
        (300, 430),
        (270, 560),
        (255, 255, 255),
        20
    )

    cv2.line(
        dummy,
        (340, 430),
        (370, 560),
        (255, 255, 255),
        20
    )

    return dummy
