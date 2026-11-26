import torch
from mmaction.models import GCNHead
from mmaction.apis import init_recognizer


def load_stgcn(cfg):
    device = cfg["runtime"]["device"]

    model = init_recognizer(
        cfg["gcn"]["config_path"],
        checkpoint=None,
        device=device,
    )

    model.cls_head = GCNHead(
        num_classes=2,
        in_channels=256,
    ).to(device)

    state_dict = torch.load(
        cfg["gcn"]["checkpoint_path"],
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model

@torch.no_grad()
def extract_feature(model, data, device):
    x = torch.from_numpy(data).float().to(device)
    x = x.unsqueeze(0).unsqueeze(0)

    feat, _ = model.extract_feat(
        x,
        stage="backbone"
    )

    return feat