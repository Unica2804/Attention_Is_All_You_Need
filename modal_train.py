import modal

volume = modal.Volume("poetry-transformer-volume",create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "datasets",
        "tokenizers",
        "tqdm",
        "tensorboard"
    )
    .add_local_file("config.py","/root/config.py")
    .add_local_file("dataset.py","/root/dataset.py")
    .add_local_file("model.py","/root/model.py")
    .add_local_file("train.py","/root/train.py")
)

app = modal.App("poetry-transformer",image=image)

@app.function(
    gpu="",
    volumes={"/root/weights": volume},
    timeout=60*60*3
)
def run_training():
    import os
    import sys
    sys.path.append("/root")
    from config import get_config
    from train import train_model
    config = get_config()

    config["model_folder"] = "/root/weights"
    config["tokenizer_filename"] = "/root/weights/tokenizer.json"
    print("Training configuration:",config)
    print("Starting training...")
    train_model(config)
    volume.commit()
    print("Training completed and weights saved to Modal volume.")

@app.local_entrypoint()
def main():
    run_training().remote()
