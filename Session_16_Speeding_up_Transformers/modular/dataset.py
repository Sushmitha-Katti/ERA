from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import datasets as data_datasets

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import pytorch_lightning as pl


class BilinualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
            ],
            dim=0,
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
            ],
            dim=0,
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


class BilinualDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def setup(self, stage):
        ds_raw = load_dataset(
            "opus_books",
            f"{self.config['lang_src']}-{self.config['lang_tgt']}",
            split="train",
        )
        # clean the raw_dataset by removing
        # english sentence whose length is > 150 tokens
        # french sentence whose length > len(english) + 10
        ds_raw = self._clean_raw_dataset(ds_raw)

        self.tokenizer_src = self.get_or_build_tokenizer(
            ds_raw, self.config["lang_src"]
        )
        self.tokenizer_tgt = self.get_or_build_tokenizer(
            ds_raw, self.config["lang_tgt"]
        )

        # devide train, test and val
        train_ds_size = int(0.8 * len(ds_raw))
        test_ds_size = int(0.1 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size - test_ds_size
        train_ds_raw, test_ds_raw, val_ds_raw = random_split(
            ds_raw, [train_ds_size, test_ds_size, val_ds_size]
        )

        # Determine Max Length if src and target
        max_len_src = 0
        max_len_tgt = 0

        for item in ds_raw:
            src_ids = self.tokenizer_src.encode(
                item["translation"][self.config["lang_src"]]
            ).ids
            tgt_ids = self.tokenizer_tgt.encode(
                item["translation"][self.config["lang_tgt"]]
            ).ids

            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        print(f"Max length of source sentence:: {max_len_src}")
        print(f"Max length of target sentence:: {max_len_tgt}")

        self.train_ds = BilinualDataset(
            train_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )
        self.val_ds = BilinualDataset(
            val_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )
        self.test_ds = BilinualDataset(
            test_ds_raw,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.config["lang_src"],
            self.config["lang_tgt"],
            self.config["seq_len"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer_tgt),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=self.config["num_workers"],
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer_tgt),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.config["num_workers"],
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer_tgt),
        )

    def get_or_build_tokenizer(self, ds, lang):
        tokenizer_path = Path(self.config["tokenizer_file"].format(lang))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
            )

            tokenizer.train_from_iterator(
                self.get_all_sentences(ds, lang), trainer=trainer
            )

            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    def get_all_sentences(self, ds, lang):
        for item in ds:
            yield item["translation"][lang]

    def get_tokenizers(self):
        return self.tokenizer_src, self.tokenizer_tgt

    def _clean_raw_dataset(self, raw_dataset):
        new_data_list = []
        new_index = 0
        for index, each_dataset in enumerate(raw_dataset):
            src_text = each_dataset["translation"][self.config["lang_src"]]
            tgt_text = each_dataset["translation"][self.config["lang_tgt"]]

            if len(src_text) > 150 or len(tgt_text) > len(src_text) + 10:
                continue

            new_data_list.append(
                {
                    "id": new_index,
                    "translation": {
                        self.config["lang_src"]: src_text,
                        self.config["lang_tgt"]: tgt_text,
                    },
                }
            )
            new_index += 1
        new_dataset = data_datasets.Dataset.from_list(new_data_list)
        return new_dataset


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def collate_fn(batch, tokenizer_tgt):
    """
    This function adds dynamic padding to each batch and also adds encoder mask and decoder mask for dataset.
    """
    encoder_inputs, decoder_inputs, labels, src_texts, tgt_texts = zip(*batch)
    decoder_inputs = [item["decoder_input"] for item in batch]
    encoder_inputs = [item["encoder_input"] for item in batch]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]
    labels = [item["label"] for item in batch]
    pad_token = torch.tensor(
        [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
    max_decoder_length = max(len(seq) for seq in decoder_inputs)
    max_encoder_length = max(len(seq) for seq in encoder_inputs)
    padded_decoder_inputs = []
    padded_encoder_inputs = []
    padded_label_inputs = []
    encoder_masks = []
    decoder_masks = []

    for decoder, encoder, label in zip(decoder_inputs, encoder_inputs, labels):
        decoder_padding_length = max_decoder_length - len(decoder)
        encoder_padding_length = max_encoder_length - len(encoder)

        # Encoder Input
        encoder_input = torch.cat(
            [
                encoder,
                torch.tensor([pad_token] * encoder_padding_length,
                             dtype=torch.int64),
            ],
            dim=0,
        )
        padded_encoder_inputs.append(encoder_input)

        # Decoder Input
        decoder_input = torch.cat(
            [
                decoder,
                torch.tensor([pad_token] * decoder_padding_length,
                             dtype=torch.int64),
            ],
            dim=0,
        )
        padded_decoder_inputs.append(decoder_input)

        # Label
        label_input = torch.cat(
            [
                label,
                torch.tensor([pad_token] * decoder_padding_length,
                             dtype=torch.int64),
            ],
            dim=0,
        )
        padded_label_inputs.append(label_input)

        # Encoder Mask
        encoders_mask = (encoder_input != pad_token).unsqueeze(
            0).unsqueeze(0).int()

        encoder_masks.append(encoders_mask)

        # Decoder Mask
        decoder_mask = (decoder_input != pad_token).unsqueeze(0).int() & causal_mask(
            decoder_input.size(0)
        )

        decoder_masks.append(decoder_mask)

    return {
        "encoder_input": torch.stack(padded_encoder_inputs),
        "decoder_input": torch.stack(padded_decoder_inputs),
        "encoder_mask": torch.stack(encoder_masks),
        "decoder_mask": torch.stack(decoder_masks),
        "label": torch.stack(padded_label_inputs),
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }
