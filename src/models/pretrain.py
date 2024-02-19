import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pathlib import Path
import logging

"""Custom code"""
from src.transformer.transformer_utils import ReZero
from src.transformer.transformer import Transformer, MaskedLanguageModel, SOP_Decoder

log = logging.getLogger(__name__)


class TransformerEncoder(pl.LightningModule):
    """Transformer with Masked Language Model and Sentence Order Prediction"""

    def __init__(self, hparams):
        super(TransformerEncoder, self).__init__()
        self.hparams.update(hparams)
        self.last_global_step = 0
        # 1. ENCODER
        self.transformer = Transformer(self.hparams)

        # 2. DECODER BLOCK
        self.task = self.hparams.training_task
        log.info("Training task: %s" % self.task)
        if "mlm" in self.task:
            # Number of outputs (for logging purposes)
            self.num_outputs = self.hparams.vocab_size

            # 2.1. DECODERS
            self.mlm_decoder = MaskedLanguageModel(
                self.hparams, self.transformer.embedding, act="tanh")
            self.sop_decoder = SOP_Decoder(self.hparams)
            # 2.2. LOSS
            # Weighting for the loss functions
            self.register_buffer("sop_weight", torch.tensor(0.2))
            self.register_buffer("mlm_weight", torch.tensor(0.8))
            self.register_buffer("sop_class_weight",
                                 torch.tensor([1/0.8, 1/0.1, 1/0.1]))
            # Loss functions
            self.sop_loss = nn.CrossEntropyLoss(
                weight=self.sop_class_weight, label_smoothing=0.1)
            self.mlm_loss = nn.CrossEntropyLoss(ignore_index=0)
        else:
            raise NotImplementedError()

    def forward(self, batch):
        """Forward pass that returns the logits for the masked language model and the sequence order prediction task."""
        # 1. ENCODER INPUT
        predicted = self.transformer(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long()
        )
        # 2. MASKED LANGUAGE MODEL
        mlm_pred = self.mlm_decoder(predicted, batch)
        # 3. SEQUENCE ORDER PREDICTION Task
        # Embedding of the CLS token
        sop_pred = self.sop_decoder(predicted[:, 0])

        return mlm_pred, sop_pred

    def calculate_total_loss(self, mlm_preds, sop_preds, batch):
        mlm_targs = batch["target_tokens"].long()
        sop_targs = batch["target_sop"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        sop_loss = self.sop_loss(sop_preds, target=sop_targs)

        total_loss = self.sop_weight * sop_loss + self.mlm_weight * mlm_loss
        return total_loss

    def training_step(self, batch, batch_idx):
        """Training Step"""
        # 1. ENCODER-DECODER
        mlm_preds, sop_preds = self(batch)
        # 2. LOSS
        return self.calculate_total_loss(mlm_preds, sop_preds, batch)

    def on_train_epoch_end(self, output):
        """On Train Epoch End: Redraw the projection of the Attention-related matrices"""
        if self.hparams.attention_type == "performer":
            self.transformer.redraw_projection_matrix(-1)
        else:
            raise NotImplementedError(
                "We only have a Performer implementation.")

    def validation_step(self, batch, batch_idx):
        """Validation Step"""
        # 1. ENCODER-DECODER
        mlm_preds, sop_preds = self(batch)
        # 2. LOSS
        return self.calculate_total_loss(mlm_preds, sop_preds, batch)

    def test_step(self, batch, batch_idx):
        # 1. ENCODER-DECODER
        mlm_preds, sop_preds = self(batch)
        # 2. LOSS
        return self.calculate_total_loss(mlm_preds, sop_preds, batch)

    def configure_optimizers(self):
        """Configuration of the Optimizer and the Learning Rate Scheduler."""
        no_decay = [
            "bias",
            "norm",
            "age",
            "abspos",
            "token",
            "decoder.g"
        ]

        # It is advised to avoid the decay on the embedding weights, biases of the model and values of the ReZero gates.

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.epsilon,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.hparams.learning_rate,
                    epochs=30, steps_per_epoch=375,
                    three_phase=False, pct_start=0.05, max_momentum=self.hparams.beta1,
                    div_factor=30
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }
