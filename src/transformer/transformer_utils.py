import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import logging
import torch
from torch import nn
import numpy as np

log = logging.getLogger(__name__)


def masked_sort(x, mask, dim=-1, descending=True):
    """Sort operation with the masked elements"""
    mask = mask.type(x.dtype)
    masked = torch.mul(x, mask)
    neg_inf = torch.zeros_like(x).to(
        masked.device).masked_fill(mask == 0, -math.inf)
    return torch.sort((masked + neg_inf), dim=dim, descending=descending)[0]


def masked_max(x, mask, dim=-1):
    """Max operation with the masked elements"""
    mask = mask.type(x.dtype)
    masked = torch.mul(x, mask)
    neg_inf = torch.zeros_like(x).to(
        masked.device).masked_fill(mask == 0, -math.inf)
    return torch.max((masked + neg_inf), dim=1)[0]


class Center(nn.Module):
    """Remove the mean from the embedding matrix"""

    def __init__(self,  ignore_index: torch.LongTensor, norm: bool = False, use_ignore_index: bool = True) -> None:
        super().__init__()
        self.register_buffer("norm", torch.BoolTensor([norm]))
        self.register_buffer("use_ignore_index",
                             torch.BoolTensor([use_ignore_index]))

        self.register_buffer("ignore_index", ignore_index)

    def forward(self, X):
        if self.use_ignore_index:
            mask = self.mask(X)
            # we do not want tokens as PAD and PLACEHOLDERS to contribute to the mean
            X = X - X[mask].mean(0)
            # we do not want to do anything with the indexes we ignore, like PAD and PLACEHOLDERS
            X[self.ignore_index] *= 0
        else:
            X = X - X.mean(0)
        if self.norm:
            return l2_norm(X)
        return X

    def mask(self, X):
        mask = torch.ones(X.shape[0])
        mask[self.ignore_index] = 0
        return mask.bool()


def cosine_annealing(current_step):
    """Cosine Annealing for the Learning Rate"""
    progress = min(current_step * 0.033, 0.95)
    return math.cos(0.5 * math.pi * progress)


#######################
# Activation Functions
#######################

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """ ""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """ ""
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "gelu_custom": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_google": gelu_new,
}

###################################
# Normalisation and Residuals
###################################


class Swish(nn.Module):
    def forward(self, input: Tensor):
        return swish(input)


def l2_norm(x):
    return F.normalize(x, dim=-1, p=2)


class Norm(nn.Module):
    def forward(self, X):
        return l2_norm(X)


class ReZero(torch.nn.Module):
    """Implementation of ReZero (Residual Connection)"""

    def __init__(self, hidden_size, simple: bool = True, fill: float = .0):
        """"""
        super(ReZero, self).__init__()
        if simple:  # aka original
            self.weights = torch.nn.Parameter(torch.add(torch.zeros(1), fill))
        else:
            self.weights = torch.nn.Parameter(
                torch.add(torch.zeros(hidden_size), fill))

    def forward(self, x, y):
        return x + y * self.weights


class ScaleNorm(torch.nn.Module):
    """L2-norm (Alternative to LayerNorm)"""

    def __init__(self, hidden_size, eps=1e-6):
        """"""
        super(ScaleNorm, self).__init__()
        self.g = torch.nn.Parameter(torch.sqrt(torch.Tensor([hidden_size])))
        self.eps = eps

    def forward(self, x):
        """"""
        norm = self.g / torch.linalg.norm(x, dim=-1, ord=2, keepdim=True).clamp(
            min=self.eps
        )
        return x * norm


class FixNorm(ScaleNorm):
    """Scale Norm with fixed weight (no gradient)"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.g.weight = torch.Tensor([1.])
        self.g.requires_grad = False


class SigSoftmax(nn.Module):
    """Implementation of SigSoftmax (prevents oversaturation)"""

    def __init__(self, dim: int = -1, epsilon: float = 1e-12):
        super().__init__()
        self.epsilon = epsilon
        self.sigmoid = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask, -np.inf)
        return self.softmax(x + torch.log(torch.sigmoid(x) + self.epsilon))
