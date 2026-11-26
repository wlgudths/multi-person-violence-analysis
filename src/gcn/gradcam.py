"""
Compute Fight Grad-CAM and interpolate its temporal dimension.
"""


import torch
import numpy as np
import torch.nn.functional as F


def build_input(window, device):
    x = torch.from_numpy(window["data"]).float().to(device)
    return x.unsqueeze(0).unsqueeze(0)

def forward_gradcam(model, x, fight_class):
    model.zero_grad(set_to_none=True)

    feat, _ = model.extract_feat(x, stage="backbone")
    feat.retain_grad()

    logits = model.cls_head(feat)
    probs = torch.softmax(logits, dim=1)

    logits[0, fight_class].backward()

    grads = feat.grad
    weights = grads.mean(dim=(3, 4), keepdim=True)

    cam = F.relu(
        (weights * feat).sum(dim=2)
    )[0].detach().cpu().numpy()

    max_value = float(cam.max())

    if max_value > 0:
        cam /= max_value

    return feat.detach(), logits.detach(), probs.detach(), cam

def upsample_cam(cam, target_T):
    M, src_T, V = cam.shape

    src_idx = np.arange(src_T)
    dst_idx = np.linspace(0, src_T - 1, target_T)

    out = np.zeros((M, target_T, V), dtype=np.float32)

    for m in range(M):
        for v in range(V):
            out[m, :, v] = np.interp(
                dst_idx, src_idx, cam[m, :, v]
            )

    return out