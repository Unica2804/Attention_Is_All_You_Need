import torch
from torch.utils.data import Dataset

def create_casual_mask(seq_len:int)->torch.Tensor:
    mask = torch.tril(torch.ones((1, seq_len, seq_len), dtype=torch.bool),diagonal=1)
    return ~mask

class PoetryDataset(Dataset):
    def __init__(self,ds,tokenizer_src,src_seq_len:int,tgt_seq_len:int):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.sos_id = tokenizer_src.token_to_id("[SOS]")
        self.eos_id = tokenizer_src.token_to_id("[EOS]")
        self.pad_id = tokenizer_src.token_to_id("[PAD]")
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        row = self.ds[idx]
        theme = row["type"] if row["type"] else "General"
        title = row["poem name"] if row["poem name"] else "Untitled"
        tgt_text = row["content"] if row["content"] else ""
        src_text = f"Theme: {theme} Title: {title}"
        src_token_ids = self.tokenizer_src.encode(src_text).ids
        tgt_token_ids = self.tokenizer_src.encode(tgt_text).ids

        if len(src_token_ids) > self.src_seq_len:
            src_token_ids = src_token_ids[:self.src_seq_len]
        
        if len(tgt_token_ids) > self.tgt_seq_len-1:
            tgt_token_ids = tgt_token_ids[:self.tgt_seq_len-1]
        
        src_pad_len = self.src_seq_len - len(src_token_ids)
        tgt_pad_len = self.tgt_seq_len - 1 - len(tgt_token_ids)

        encoder_input = torch.tensor(
            src_token_ids + [self.pad_id] * src_pad_len,
            dtype=torch.long
        )

        decoder_input = torch.tensor(
            [self.sos_id] + tgt_token_ids + [self.pad_id] * tgt_pad_len,
            dtype=torch.long
        )

        label = torch.tensor(
            tgt_token_ids + [self.eos_id] + [self.pad_id] * tgt_pad_len,
            dtype=torch.long
        )
        
        src_mask = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0).int()
        decoder_pad_mask = (decoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)
        casual_mask = create_casual_mask(self.tgt_seq_len)
        tgt_mask = (decoder_pad_mask & casual_mask).int()

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
        