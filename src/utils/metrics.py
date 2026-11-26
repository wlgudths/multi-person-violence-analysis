import numpy as np


def softmax(logits, axis=-1):
    logits = np.asarray(logits)
    logits = logits - np.max(logits, axis=axis, keepdims=True)

    exp_logits = np.exp(logits)

    return exp_logits / exp_logits.sum(axis=axis, keepdims=True)

def logit_margin(logits, positive_class, negative_class):
    logits = np.asarray(logits)

    return logits[..., positive_class] - logits[..., negative_class]