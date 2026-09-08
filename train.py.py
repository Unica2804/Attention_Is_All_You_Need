import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from pathlib import Path
from dataset import PoetryDataset
from model import make_model

def dataset_iterator(dataset, batch_size:int = 1000):
    for i in range(0, len(dataset), batch_size):
        batch=dataset[i:i+batch_size]
        texts = []
        for genre, title, content in zip(batch["type"],batch["poem name"],batch["content"]):
            genre_str = genre if genre else "General"
            title_str = title if title else "Untitled"
            content_str = content if content else ""
            formatted = f"Theme: {genre_str} Title: {title_str} Content: {content_str}"
            texts.append(formatted)
        yield texts

def get_build_tokenizer(config:dict,dataset=None)->Tokenizer:
    tokenizer_path = Path(config["tokenizer_path"])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],min_frequency=2)
        
        if dataset is not None:
            iterator = dataset_iterator(dataset)
            tokenizer.train_from_iterator(iterator, trainer)
        else:
            tokenizer.train([config["train_data_path"]], trainer)
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_datasets(config:dict):
    ds_raw = load_dataset("merve/poetry",split="train")
    tokenizer_src = get_build_tokenizer(config,ds_raw)
    train_ds_size = int(len(ds_raw)*0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    ds_train_raw, ds_val_raw = torch.utils.data.random_split(ds_raw,[train_ds_size,val_ds_size])
    train_ds = PoetryDataset(ds_train_raw,tokenizer_src,config["src_seq_len"],config["tgt_seq_len"])
    val_ds = PoetryDataset(ds_val_raw,tokenizer_src,config["src_seq_len"],config["tgt_seq_len"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, tokenizer_src

def train_model(config:dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, tokenizer_src = get_datasets(config)

    vocab_size = tokenizer_src.get_vocab_size()
    model = make_model(
        
    )