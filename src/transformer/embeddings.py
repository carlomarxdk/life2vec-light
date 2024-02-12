import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from src.transformer.transformer_utils import ReZero, Center

import logging
log = logging.getLogger(__name__)


class Embeddings(nn.Module):
    """Class for token, position, segment and backgound embedding."""

    def __init__(self, hparams):
        super(Embeddings, self).__init__()
        embedding_size = hparams.hidden_size

        # Initialize Token/Concept embedding matrix
        self.token = nn.Embedding(
            hparams.vocab_size,
            embedding_size,
            padding_idx=0
        )
        # Initialize Segment embedding matrix
        self.segment = nn.Embedding(4, hparams.hidden_size, padding_idx=0)

        # Initialize Time2Vec embeddings
        self.age = PositionalEmbedding(1, hparams.hidden_size, torch.cos)
        self.abspos = PositionalEmbedding(1, hparams.hidden_size, torch.sin)

        # Uniformly initialise the weights of the embedding matrix
        d = 0.01
        nn.init.uniform_(self.token.weight, a=-d, b=d)
        nn.init.uniform_(self.segment.weight, a=-d, b=d)

        if hparams.parametrize_emb:
            try:
                # The centering of the embedding matrix
                self.parametrize(norm=hparams.norm_input_emb)
            except:
                log.info(
                    "(EMBEDDING) Normalisation hyperparameter is not found, set to FALSE")
                self.parametrize()

        self.res_age = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.res_abs = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.res_seg = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.dropout = nn.Dropout(hparams.emb_dropout)

    def parametrize(self, norm: bool = False):
        """Remove Mean from the Embedding Matrix (on each forward pass"""
        ignore_index = torch.LongTensor(
            [0, 4, 5, 6, 7, 8])  # We ignore the 5 tokens (PAD, and placeholder tokens that we included into the language but never used)

        parametrize.register_parametrization(
            self.token, "weight", Center(ignore_index=ignore_index, norm=norm))

    def reparametrization(self):
        """Remove the parametrization from the Concept Embedding Matrix"""

        parametrize.remove_parametrizations(
            self.token, "weight", leave_parametrized=False)

    def forward(self, tokens, position, age, segment):
        """"""
        tokens = self.token(tokens)

        pos = self.age(age.float().unsqueeze(-1))
        pos[:, :5] *= 0
        tokens = self.res_age(tokens, pos)

        pos = self.abspos(position.float().unsqueeze(-1))
        pos[:, :5] *= 0
        tokens = self.res_abs(tokens, pos)

        if segment is not None:
            pos = self.segment(segment)
            tokens = self.res_seg(tokens, pos)

        return self.dropout(tokens), None


# TIME2VEC IMPLEMENTATION

def t2v(tau, f, w, b, w0, b0, arg=None):
    """Time2Vec function"""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class PositionalEmbedding(nn.Module):
    """Implementation of Time2Vec"""

    def __init__(self, in_features, out_features, f):
        super(PositionalEmbedding, self).__init__()

        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1))
        self.f = f

        d = 0.01
        nn.init.uniform_(self.w0, a=-d, b=d)
        nn.init.uniform_(self.b0, a=-d, b=d)
        nn.init.uniform_(self.w, a=-d, b=d)
        nn.init.uniform_(self.b, a=-d, b=d)

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)
