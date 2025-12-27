import cv2
import numpy as np
import torch

def compute_optical_flow(frames):
    flows = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

    for i in range(1, len(frames)):
        curr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Convert flow to magnitude image
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flows.append(mag.astype(np.uint8))

        prev = curr

    return flows
