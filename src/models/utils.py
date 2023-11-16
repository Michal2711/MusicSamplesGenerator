import torch
import torch.nn as nn
from math import inf

# PixelNorm - instead of BatchNorm
# we are getting some pixel values and normalize it by dividing it by mean of pixel values square
# n - represent number of channels so for spectrograms is useless

# Equalize Learning Rate
# for each forward step we take the weight we have and multiply by scale value
# scale value = sqrt(2/(k*k*c) : k - kernel size, c - input channels)

class AudioNorm(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)

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

def miniBatchStdDev(x):
    batch_statistics = (
        torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
    )
    return torch.cat([x, batch_statistics], dim=1)

def WGANGPGradientPenalty(input, fake, discriminator, weight, backward=True):
    r"""
    Gradient penalty as described in
    "Improved Training of Wasserstein GANs"
    https://arxiv.org/pdf/1704.00028.pdf

    Args:

        - input (Tensor): batch of real data
        - fake (Tensor): batch of generated data. Must have the same size
          as the input
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """

    batchSize = input.size(0)
    alpha = torch.rand(batchSize, 1)

    alpha = alpha.expand(batchSize, int(input.nelement() /
                                        batchSize)).contiguous().view(
                                            input.size())

    alpha = alpha.to(input.device)

    interpolates = alpha * input + ((1 - alpha) * fake)

    interpolates = torch.autograd.Variable(
        interpolates, requires_grad=True)

    decisionInterpolate = discriminator(interpolates, False)

    # We get the last element, the rest are the att predictions
    decisionInterpolate = decisionInterpolate[:, -1].sum()
    # decisionInterpolate = decisionInterpolate[:, 0].sum()


    gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                    inputs=interpolates,
                                    create_graph=True, retain_graph=True)

    # gradients = gradients[0].view(batchSize, -1).norm(2, dim=1)
    gradients = gradients[0].view(batchSize, -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()

    gradient_penalty = (((gradients - 1.0)**2)).sum() * weight
    # gradient_penalty = (((gradients - 1.0)**2)).mean() * weight
    lipschitz_norm = gradients.max()

    if backward:
        try:
            # This is necessary for execution with CPU. Not sure yet why...
            gradient_penalty.requires_grad = True
        except RuntimeError as e:
            pass
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item(), lipschitz_norm.item()


# simplified solution from DrumGAN model
# def miniBatchStdDev(x, subGroupSize=4):
#     r"""
#     Add a minibatch standard deviation channel to the current layer.
#     In other words:
#         1) Compute the standard deviation of the feature map over the minibatch
#         2) Get the mean, over all pixels and all channels of thsi ValueError
#         3) expand the layer and cocatenate it with the input

#     Args:

#         - x (tensor): previous layer
#         - subGroupSize (int): size of the mini-batches on which the standard deviation
#         should be computed
#     """
#     size = x.size()
#     nDim = len(size)
#     subGroupSize = min(size[0], subGroupSize)
#     if size[0] % subGroupSize != 0:
#         subGroupSize = size[0]
#     G = int(size[0] / subGroupSize)
#     if subGroupSize > 1:
#         if nDim == 3: y = x.view(subGroupSize, -1, size[1], size[2])
#         else: y = x.view(subGroupSize, -1, size[1], size[2], size[3])
#         y = torch.var(y, 0)
#         y = torch.sqrt(y + 1e-8)
#         y = y.view(G, -1)
#         y = torch.mean(y, 1).view(G, 1)
        
#         if nDim == 3:
#             y = y.expand(G, size[2]).view((G, 1, 1, size[2]))
#             y = y.expand(G, subGroupSize, -1, -1)
#             y = y.contiguous().view((-1, 1, size[2]))
#         else: 
#             y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
#             y = y.expand(G, subGroupSize, -1, -1, -1)
#             y = y.contiguous().view((-1, 1, size[2], size[3]))
#     else:
        
#         if nDim == 3: y = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
#         else: y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

#     output = torch.cat([x, y], dim=1)
#     return output
