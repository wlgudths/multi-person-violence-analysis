import torch
from gcn.gcn import extract_feature


def classify(model, features, fight_class):
    logits = model.cls_head(features)
    probs = torch.softmax(logits, dim=1)

    return {
        "pred": int(probs.argmax(dim=1).item()),
        "logits": logits[0].detach().cpu().tolist(),
        "probs": probs[0].detach().cpu().tolist(),
        "fight_prob": float(probs[0, fight_class]),
    }

def classify_persons(model, features, track_ids, fight_class):
    persons = []

    for i, track_id in enumerate(track_ids):
        result = classify(
            model,
            features[:, i:i + 1],
            fight_class,
        )

        persons.append({
            "track_id": int(track_id),
            "logits": result["logits"],
            "probs": result["probs"],
            "fight_prob": result["fight_prob"],
        })

    return persons

@torch.no_grad()
def infer_window(model, window, device, fight_class):
    features = extract_feature(
        model,
        window["data"],
        device,
    )

    all_result = classify(
        model,
        features,
        fight_class,
    )

    person_results = classify_persons(
        model,
        features,
        window["track_ids"],
        fight_class,
    )

    return {
        **all_result,
        "persons": person_results,
    }