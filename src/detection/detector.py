import numpy as np
from mmcv.transforms import Compose
from mmengine.registry import DefaultScope
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import get_test_pipeline_cfg


def load_detector(cfg):
    det_cfg = cfg["detection"]

    model = init_detector(
        det_cfg["config_path"],
        det_cfg["checkpoint_path"],
        device=cfg["runtime"]["device"],
    )

    model.eval()

    with DefaultScope.overwrite_default_scope("mmdet"):
        test_pipeline = get_test_pipeline_cfg(model.cfg.copy())

        test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

        model._test_pipeline = Compose(test_pipeline)

    return model

def detect_persons(model, frame, score_thr=0.3):

    result = inference_detector(model, frame, test_pipeline=model._test_pipeline)

    pred = result.pred_instances

    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()

    # COCO person class = 0
    keep = (labels == 0) & (scores >= score_thr)

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(bboxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    detections = np.concatenate(
        [
            bboxes,
            scores[:, None],
            labels[:, None]
        ],
        axis=1
    )
    
    return detections.astype(np.float32)

