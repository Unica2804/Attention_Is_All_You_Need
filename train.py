import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import warnings

from dataset import PoetryDataset
from model import make_model
from config import get_config, get_weights_path


def dataset_iterator(dataset, batch_size:int = 1000):
    for i in range(0, len(dataset), batch_size):
        batch=dataset[i:i+batch_size]
        texts = []
        for genre, title, content in zip(batch["type"],batch["poem name"],batch["content"]):
            genre_str = genre if genre else "General"
            title_str = title if title else "Untitled"
            content_str = content if content else ""
            src_prompt = f"Theme: {genre_str} Title: {title_str}"
            texts.append(src_prompt)
            texts.append(content_str)
        yield texts

def get_build_tokenizer(config:dict,dataset=None)->Tokenizer:
    tokenizer_path = Path(config["tokenizer_filename"])
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
    print(f"Using device: {device}")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, tokenizer_src = get_datasets(config)

    vocab_size = tokenizer_src.get_vocab_size()
    model = make_model(
        vocab_size,
        vocab_size,
        config["src_seq_len"],
        config["tgt_seq_len"]
    ).to(device)
    writer=SummaryWriter(log_dir=config["experiment_name"])
    
    optim = torch.optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9,betas=(0.9,0.98))

    init_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_path(config,config["preload"])
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename)
        init_epoch = state["epoch"]+1
        optim.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(init_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in batch_iterator:
            optim.zero_grad()
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)

            encoder_output = model.encode(encoder_input, src_mask)
            decoder_output = model.decode(decoder_input,encoder_output, src_mask, tgt_mask)
            projected_output = model.project(decoder_output)
            label = batch["label"].to(device)
            loss = loss_fn(projected_output.view(-1,tokenizer_src.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            optim.step()
            
            global_step += 1
        model_filename = get_weights_path(config, f"{epoch:02d}")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'global_step': global_step
            },model_filename
        )
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)