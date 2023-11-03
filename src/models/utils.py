import torch
from math import inf

def isinf(tensor):
    r"""Returns a new tensor with boolean elements representing if each element
    is `+/-INF` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of
        `+/-INF` elements and 0 otherwise

    Example::

        >>> torch.isinf(torch.Tensor([1, float('inf'), 2,
                            float('-inf'), float('nan')]))
        tensor([ 0,  1,  0,  1,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor.abs() == inf

def isnan(tensor):
    r"""Returns a new tensor with boolean elements representing if each element
    is `NaN` or not.

    Arguments:
        tensor (Tensor): A tensor to check

    Returns:
        Tensor: A ``torch.ByteTensor`` containing a 1 at each location of `NaN`
        elements.

    Example::

        >>> torch.isnan(torch.tensor([1, float('nan'), 2]))
        tensor([ 0,  1,  0], dtype=torch.uint8)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The argument is not a tensor", str(tensor))
    return tensor != tensor

def finiteCheck(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    for p in parameters:
        infGrads = isinf(p.grad.data)
        p.grad.data[infGrads] = 0

        nanGrads = isnan(p.grad.data)
        p.grad.data[nanGrads] = 0

def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size, channels, height, width = real.shape
    eps = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolated_images = real * eps + fake * (1 - eps)

    # critic scores
    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

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
