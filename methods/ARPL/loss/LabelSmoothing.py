import torch

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):

    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1

    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

def smooth_cross_entropy_loss(logits, labels, smoothing, dim=-1):

    """
    :param logits: Predictions from model (before softmax) (B x C)
    :param labels: LongTensor of class indices (B,)
    :param smoothing: Float, how much label smoothing
    :param dim: Channel dimension
    :return:
    """

    # Convert labels to distributions
    labels = smooth_one_hot(true_labels=labels, smoothing=smoothing, classes=logits.size(dim))

    preds = logits.log_softmax(dim=dim)

    return torch.mean(torch.sum(-labels * preds, dim=dim))