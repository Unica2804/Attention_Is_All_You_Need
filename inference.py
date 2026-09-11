import os
import torch
from pathlib import Path
from tokenizers import Tokenizer

from config import get_config, get_weights_path
from model import make_model
from dataset import create_casual_mask
from train import get_datasets
import torchmetrics


def greedy_decode(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    tokenizer: Tokenizer,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    sos_idx = tokenizer.token_to_id("[SOS]")
    eos_idx = tokenizer.token_to_id("[EOS]")

    # 1. Encode prompt once
    encoder_output = model.encode(src, src_mask)

    # 2. Seed decoder input with [SOS]
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Shape: (1, current_len, current_len)
        decoder_mask = create_casual_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        # Forward pass through decoder and linear projection
        out = model.decode(decoder_input, encoder_output, src_mask, decoder_mask)
        prob = model.project(out[:, -1])  # (1, vocab_size)

        _, next_word = torch.max(prob, dim=1)

        # Append predicted token ID
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device)],
            dim=1,
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def generate_from_prompt(
    model,
    tokenizer: Tokenizer,
    theme: str,
    title: str,
    src_seq_len: int,
    max_len: int,
    device: torch.device,
) -> str:
    model.eval()

    pad_idx = tokenizer.token_to_id("[PAD]")
    src_text = f"Theme: {theme} Title: {title}"
    src_token_ids = tokenizer.encode(src_text).ids

    # Truncate if longer than configured sequence length
    if len(src_token_ids) > src_seq_len:
        src_token_ids = src_token_ids[:src_seq_len]

    # Pad to match fixed encoder positional encoding shape
    src_pad_len = src_seq_len - len(src_token_ids)
    encoder_input = torch.tensor(
        src_token_ids + [pad_idx] * src_pad_len, dtype=torch.long
    ).unsqueeze(0).to(device)  # (1, src_seq_len)

    src_mask = (encoder_input != pad_idx).unsqueeze(0).unsqueeze(0).int().to(device)  # (1, 1, 1, src_seq_len)

    with torch.no_grad():
        output_tokens = greedy_decode(model, encoder_input, src_mask, tokenizer, max_len, device)

    return tokenizer.decode(output_tokens.detach().cpu().numpy().tolist())

def validate_against_val_set(
    model,
    val_loader,
    tokenizer: Tokenizer,
    max_len: int,
    device: torch.device,
    num_samples: int = 5,
    compute_metrics: bool = True,
):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        console_width = 80

    print("\n" + "=" * console_width)
    print("RUNNING VALIDATION EVALUATION")
    print("=" * console_width)

    with torch.no_grad():
        for batch in val_loader:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["src_mask"].to(device)

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer.decode(model_out.detach().cpu().numpy().tolist())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print("-" * console_width)
            print(f"{'SOURCE:':>12} {source_text}")
            print(f"{'TARGET:':>12} {target_text}")
            print(f"{'PREDICTED:':>12} {model_out_text}")

            if count >= num_samples:
                print("-" * console_width)
                break

    if compute_metrics and len(predicted) > 0:
        cer_metric = torchmetrics.text.CharErrorRate()
        wer_metric = torchmetrics.text.WordErrorRate()
        bleu_metric = torchmetrics.text.BLEUScore()

        cer = cer_metric(predicted, expected).item()
        wer = wer_metric(predicted, expected).item()
        bleu = bleu_metric(predicted, [[exp] for exp in expected]).item()

        print("\n" + "=" * console_width)
        print("AGGREGATE METRICS")
        print(f"  BLEU Score: {bleu:.4f}")
        print(f"  Word Error Rate (WER): {wer:.4f}")
        print(f"  Character Error Rate (CER): {cer:.4f}")
        print("=" * console_width + "\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    # _, val_loader,tokenizer_1 = get_datasets(config)

    tokenizer_path = config["tokenizer_filename"]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    model = make_model(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=config["src_seq_len"],
        tgt_seq_len=config["tgt_seq_len"],
        d_model=config["d_model"],
    ).to(device)

    # Load checkpoint (specify epoch, e.g. "01")
    epoch_to_load = "199"
    model_filename = get_weights_path(config, epoch_to_load)
    print(f"Loading weights from {model_filename}")

    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    # Inference test
    theme = "Sad"
    title = "The Lonely Night"
    poem = generate_from_prompt(
        model=model,
        tokenizer=tokenizer,
        theme=theme,
        title=title,
        src_seq_len=config["src_seq_len"],
        max_len=config["tgt_seq_len"],
        device=device,
    )

    print("\n" + "=" * 50)
    print(f"Theme: {theme} | Title: {title}\n")
    print(poem)
    print("=" * 50)
    # validate_against_val_set(
    #     model=model,
    #     val_loader=val_loader,
    #     tokenizer=tokenizer,
    #     max_len=config["tgt_seq_len"],
    #     device=device,
    #     num_samples=10,
    #     compute_metrics=True,
    # )


if __name__ == "__main__":
    main()