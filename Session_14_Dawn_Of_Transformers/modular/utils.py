import os

import torch
import torchmetrics

from dataset import causal_mask
from model import build_transformer

def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen("stty size", r) as console:
            _, console_width = console.read().split()
            console_width = int(console_width)

    except:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            print_msg("_" * console_width)
            print_msg(f"{f'SOURCE:':>12}{source_text}")
            print_msg(f"{f'TARGET:':>12}{target_text}")

            print_msg(f"{f'PREDICTED:':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

            if writer:
                metric = torchmetrics.CharErrorRate()
                cer = metric(predicted, expected)
                writer("validation cer", cer)
                # writer.flush()

                metric = torchmetrics.WordErrorRate()
                wer = metric(predicted, expected)
                writer("validation wer", wer)
                # writer.flush()

                metric = torchmetrics.BLEUScore()
                bleu = metric(predicted, expected)
                writer("validation BLEU", bleu)
                # writer.flush()


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    )
    return model


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # pre compute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calcualte output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)
