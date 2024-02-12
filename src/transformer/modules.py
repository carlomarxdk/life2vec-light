import torch.nn as nn
import time
from src.transformer.attention import MultiHeadAttention
from src.transformer.transformer_utils import ACT2FN, ReZero
import logging

log = logging.getLogger(__name__)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by layer normalisation
    """

    def __init__(
        self, hparams
    ):
        """"""
        super(SublayerConnection, self).__init__()
        assert hparams.norm_type in ["pre_norm", "rezero"]

        self.norm_type = hparams.norm_type
        hidden_size = hparams.hidden_size

        if self.norm_type == "rezero":
            self.norm = ReZero(hidden_size)
        elif self.norm_type == "pre_norm":
            raise NotImplementedError("PRE NORM is not implemented")

    def forward(self, x, sublayer, **kwargs):
        """
        Apply a residual connection to a sublayer"
        """
        if self.norm_type == "rezero":
            """
            ReZero
            """
            return self.norm(x, sublayer(x, **kwargs))
        elif self.norm_type == "pre_norm":
            """
            PRE NORM (ScaleNorm + Gate)
            """
            return self.gate(x, sublayer(self.norm(x), **kwargs))


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    """

    def __init__(self, hparams):
        """"""
        super(PositionWiseFeedForward, self).__init__()
        self.hidden2ff = nn.Linear(hparams.hidden_size, hparams.hidden_ff)
        self.dropout = nn.Dropout(hparams.fw_dropout)
        self.act = ACT2FN[hparams.hidden_act]
        self.ff2hidden = nn.Linear(hparams.hidden_ff, hparams.hidden_size)

    def forward(self, x):
        """"""
        x = self.act(self.dropout(self.hidden2ff(x)))
        return self.ff2hidden(x)


class EncoderLayer(nn.Module):
    """Encoder Block"""

    def __init__(self, hparams):
        """"""
        super(EncoderLayer, self).__init__()

        assert (
            hparams.hidden_size % hparams.n_heads == 0
        ), "Encoder: Incorrect hidden_size (%s, %s)" % (
            hparams.hidden_size,
            hparams.n_heads,
        )
        start = time.time()

        self.attention = MultiHeadAttention(hparams)
        self.attention_sublayer = SublayerConnection(hparams)

        self.position_wise = PositionWiseFeedForward(hparams)
        self.position_sublayer = SublayerConnection(hparams)

        log.info("EncoderLayer setup is finised:  %.3f s" %
                 (time.time() - start))

    def redraw_projection_matrix(self):
        """Redraw projection matrices during the training"""
        try:
            try:
                self.attention.attention.fast_attention.redraw_projection_matrix(
                    "cuda")
            except:
                self.attention.attention.fast_attention.redraw_projection_matrix(
                    "cpu")
        except:
            log.warning(
                "Cannot redraw random projections. Wrong attention type")

    def forward(self, x, mask=None):
        """Forward Pass"""
        x = self.attention_sublayer(x, sublayer=self.attention, mask=mask)
        x = self.position_sublayer(x, sublayer=self.position_wise)

        return x
