import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from src.transformer.embeddings import Embeddings
from src.transformer.transformer_utils import ScaleNorm, l2_norm, Center, Swish, ACT2FN
from src.transformer.modules import EncoderLayer
import logging

log = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(self, hparams):
        """Encoder part of the life2vec model"""
        super(Transformer, self).__init__()

        self.hparams = hparams
        # Initialize the Embedding Layer
        self.embedding = Embeddings(hparams=hparams)
        # Initialize the Encoder Blocks
        self.encoders = nn.ModuleList(
            [EncoderLayer(hparams) for _ in range(hparams.n_encoders)]
        )

    def forward(self, x, padding_mask):
        """Forward pass"""
        x, _ = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )
        for layer in self.encoders:
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)
        return x

    def forward_finetuning(self, x, padding_mask=None):

        x, _ = self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )

        for _, layer in enumerate(self.encoders):
            x = torch.einsum("bsh, bs -> bsh", x, padding_mask)
            x = layer(x, padding_mask)

        return x

    def get_sequence_embedding(self, x):
        """Get only embeddings"""
        return self.embedding(
            tokens=x[:, 0], position=x[:, 1], age=x[:, 2], segment=x[:, 3]
        )

    def redraw_projection_matrix(self, batch_idx: int):
        """Redraw projection Matrices for each layer (only valid for Performer)"""
        if batch_idx == -1:
            log.info("Redrawing projections for the encoder layers (manually)")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()

        elif batch_idx > 0 and batch_idx % self.hparams.feature_redraw_interval == 0:
            log.info("Redrawing projections for the encoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()


class MaskedLanguageModel(nn.Module):
    """Masked Language Model (MLM) Decoder (for pretraining)"""

    def __init__(self, hparams, embedding, act: str = "tanh"):
        super(MaskedLanguageModel, self).__init__()
        self.hparams = hparams
        self.act = ACT2FN[act]
        self.dropout = nn.Dropout(p=self.hparams.emb_dropout)

        self.V = nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size)
        self.g = nn.Parameter(torch.tensor([hparams.hidden_size**0.5]))
        self.out = nn.Linear(
            self.hparams.hidden_size,
            self.hparams.vocab_size,
            bias=False
        )
        if self.hparams.weight_tying == "wt":
            log.info("MLM decoder WITH Wight Tying")
            try:
                self.out.weight = embedding.token.parametrizations.weight.original
            except:
                log.warning("MLM decoder parametrization failed")
                self.out.weight = embedding.token.weight

        if self.hparams.parametrize_emb:
            ignore_index = torch.LongTensor([0, 4, 5, 6, 7, 8])
            log.info("(MLM Decoder) centering: true normalisation: %s" %
                     hparams.norm_output_emb)
            parametrize.register_parametrization(self.out, "weight", Center(
                ignore_index=ignore_index, norm=hparams.norm_output_emb))

    def batched_index_select(self, x, dim, indx):
        """Gather the embeddings of tokens that we should make prediction on"""
        indx_ = indx.unsqueeze(2).expand(
            indx.size(0), indx.size(1), x.size(-1))
        return x.gather(dim, indx_)

    def forward(self, logits, batch):
        indx = batch["target_pos"].long()
        logits = self.dropout(self.batched_index_select(logits, 1, indx))
        logits = self.dropout(l2_norm(self.act(self.V(logits))))
        return self.g * self.out(logits)


class SOP_Decoder(nn.Module):
    """Sequence Order Decoder (for pretraining)"""

    def __init__(self, hparams):
        super(SOP_Decoder, self).__init__()
        hidden_size = hparams.hidden_size
        num_targs = hparams.cls_num_targs
        p = hparams.dc_dropout

        self.in_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.act = ACT2FN["swish"]
        self.out_layer = nn.Linear(hidden_size, num_targs)

    def forward(self, x, **kwargs):
        """Foraward Pass"""
        x = self.dropout(self.norm(self.act(self.in_layer(x))))
        return self.out_layer(x)
