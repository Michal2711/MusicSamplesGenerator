import torch
import torch.nn as nn
from math import inf
import random

def init_seed(rand_seed=True):
    if not rand_seed:
        seed = random.randint(0, 9999)
    else:
        seed = 0

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rand_seed)

# simplified solution from DrumGAN model
def miniBatchStdDev(x, subGroupSize=4):
    r"""
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels of thsi ValueError
        3) expand the layer and cocatenate it with the input

    Args:

        - x (tensor): previous layer
        - subGroupSize (int): size of the mini-batches on which the standard deviation
        should be computed
    """
    size = x.size()
    nDim = len(size)
    subGroupSize = min(size[0], subGroupSize)
    if size[0] % subGroupSize != 0:
        subGroupSize = size[0]
    G = int(size[0] / subGroupSize)
    if subGroupSize > 1:
        if nDim == 3: y = x.view(subGroupSize, -1, size[1], size[2])
        else: y = x.view(subGroupSize, -1, size[1], size[2], size[3])
        y = torch.var(y, 0)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        
        if nDim == 3:
            y = y.expand(G, size[2]).view((G, 1, 1, size[2]))
            y = y.expand(G, subGroupSize, -1, -1)
            y = y.contiguous().view((-1, 1, size[2]))
        else: 
            y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
            y = y.expand(G, subGroupSize, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        
        if nDim == 3: y = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        else: y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    output = torch.cat([x, y], dim=1)
    return output
