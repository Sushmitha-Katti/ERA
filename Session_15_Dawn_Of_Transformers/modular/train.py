from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from config import get_weights_file_path
from utils import run_validation

class TextTranslator(pl.LightningModule):
    def __init__(
        self, model, tokenizer_src, tokenizer_tgt, config, validation_ds, least_loss
    ):
        super().__init__()
        self.model = model
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.config = config
        self.validation_ds = validation_ds
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )
        self.least_loss = least_loss

        self.losses = []

        Path(self.config["model_folder"]).mkdir(parents=True, exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]
        decoder_input = batch["decoder_input"]
        encoder_mask = batch["encoder_mask"]
        decoder_mask = batch["decoder_mask"]

        encoder_output = self.model.encode(encoder_input, encoder_mask)
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        proj_output = self.model.project(decoder_output)

        label = batch["label"]
        loss = self.loss_fn(
            proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )
        self.log("train_loss", loss.item(), prog_bar=True)
        self.losses.append(loss.item())

        return loss

    def on_train_epoch_end(self):
        avg_epoch_loss = np.mean(self.losses)
        print(f"Loss for epoch {self.trainer.current_epoch} is {avg_epoch_loss}")
        self.losses.clear()
        prev_loss = self.least_loss
        self.least_loss = min(avg_epoch_loss, self.least_loss)
        model_filename = get_weights_file_path(
            self.config, f"{self.trainer.current_epoch:02d}"
        )
        if prev_loss > avg_epoch_loss:
            print("Saving Model")
            torch.save(
                {
                    "epoch": self.trainer.current_epoch,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.trainer.optimizers[0].state_dict(),
                    "loss": self.least_loss,
                },
                model_filename,
            )

        run_validation(
            self.model,
            self.validation_ds,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["seq_len"],
            self.device,
            lambda msg: print(msg),
            self.global_step,
            lambda x, y: self.log(x, y),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], eps=1e-9
        )

        return [optimizer]