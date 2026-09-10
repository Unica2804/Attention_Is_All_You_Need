from pathlib import Path
def get_config():
    config = {
        "batch_size":8,
        "num_epochs": 200,
        "lr":10**-4,
        "src_seq_len": 40,
        "tgt_seq_len": 512,
        "d_model": 512,
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_filename": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel_{0}",
    }
    return config

def get_weights_path(config, epoch:str):
    model_folder = config["model_folder"]
    model_basename = config["model_filename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)