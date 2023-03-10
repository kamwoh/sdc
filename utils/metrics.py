def calculate_accuracy(logits, labels, onehot=True, multiclass=False, topk=1):
    if multiclass or topk != 1:
        if topk == 1:
            topk = 5
        pred = logits.topk(topk, 1, True, True)[1].t()
        if onehot:
            labels = labels.argmax(1)
        correct = pred.eq(labels.reshape(1, -1).expand_as(pred))
        acc = correct[:topk].reshape(-1).float().sum(0, keepdim=True) / logits.size(0)
    else:
        if len(labels.size()) == 2:
            labels = labels.argmax(1)

        acc = (logits.argmax(1) == labels).float().mean()
    return acc