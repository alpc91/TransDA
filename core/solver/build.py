import torch

def adjust_learning_rate(method, base_lr, iters, warmup_iters, warmup_ratio, max_iters, power=1.0):
    if method=='poly':
        if iters >= warmup_iters:
            lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
        else:
            k = (1 - iters / warmup_iters) * (1 - warmup_ratio)
            lr = base_lr * (1 - k)
    else:
        raise NotImplementedError
    return lr
