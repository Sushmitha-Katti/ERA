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


# def get_all_sentences(ds, lang):
#   for item in ds:

#     yield item['translation'][lang]

# def get_or_build_tokenizer(config, ds, lang):
#   tokenizer_path = Path(config['tokenizer_file'].format(lang))
#   if not Path.exists(tokenizer_path):
#     tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

#     tokenizer.pre_tokenizer = Whitespace()
#     trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)

#     tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
#     print(tokenizer)
#     tokenizer.save(str(tokenizer_path))
#   else:
#     tokenizer = Tokenizer.from_file(str(tokenizer_path))
#   return tokenizer

# def get_ds(config):

#   ds_raw = load_dataset('opus_books',f"{config['lang_src']}-{config['lang_tgt']}", split='train')
#   print(f"{config['lang_src']}-{config['lang_tgt']}")

#   tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
#   tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])


#   train_ds_size = int(0.9 * len(ds_raw))
#   val_ds_size = len(ds_raw) - train_ds_size
#   train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


#   train_ds = BilinualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
#   val_ds = BilinualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])


#   max_len_src = 0
#   max_len_tgt = 0


#   for item in ds_raw:

#     src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
#     tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids


#     max_len_src = max(max_len_src, len(src_ids))
#     max_len_tgt = max(max_len_tgt, len(tgt_ids))


#   print(f'Max length of soruce sentence:: {max_len_src}')
#   print(f'Max length of target sentence:: {max_len_tgt}')

#   train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle = True)
#   val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

#   return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


# def train_model(config):
#   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   # print("Using device: ", device)


#   # Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

#   # train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

#   # model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)


#   # Tensor board
#   # writer = SummaryWriter(config['experiment_name'])
#   # optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)

#   initial_epoch = 0
#   global_step = 0

#   if config['preload']:
#     model_filename =  get_weights_file_path(config, config['preload'])
#     print(f'Preloadiing model {model_filename}')
#     state = torch.load(model_filename)
#     model.load_state_dict(state['model_state_dict'])
#     initial_epoch = state['epoch'] + 1
#     optimizer.load_state_dict(state['optimizer_state_dict'])
#     global_step = state['gloabal_step']
#     print('preloaded')

#   # loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

#   for epoch in range(initial_epoch, config['num_epochs']):
#     torch.cuda.empty_cache()
#     model.train()

#     batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}", mininterval=5)
#     for batch in batch_iterator:
#       encoder_input = batch['encoder_input'].to(device)
#       decoder_input = batch['decoder_input'].to(device)
#       encoder_mask = batch['encoder_mask'].to(device)
#       decoder_mask = batch['decoder_mask'].to(device)

#       encoder_output = model.encode(encoder_input, encoder_mask)
#       decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
#       proj_output = model.project(decoder_output)


#       label = batch['label'].to(device)

#       loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

#       batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})


#       writer.add_scalar('train loss', loss.item(), global_step)
#       writer.flush()

#       loss.backward()
#       optimizer.step()
#       optimizer.zero_grad(set_to_none=True)

#       global_step += 1

#     run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), global_step, lambda x,y: self.log(x, y))

#     model_filename = get_weights_file_path(config, f"{epoch:02d}")

#     torch.save({
#           'epoch': epoch,
#           'model_state_dict': model.state_dict(),
#           'optimizer_state_dict': optimizer.state_dict(),
#           'gloabal_step': global_step
#       }, model_filename)
