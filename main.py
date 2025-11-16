import torch

from mlp_vae.model import train_fn
from utils import seed_everything
from submission import generate_submission

SEED = 42

CONFIG = {
    "TEXT_VE_PATH_TPL": "best_text_ve_fold_{}.pth",
    "TRANSLATOR_PATH_TPL": "best_translator_mlp_fold_{}.pth",
    "SUBMISSION_PATH": "submission_mlpvae.csv",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "EPOCHS": 150,
    "PATIENCE": 30,
    "N_FOLDS": 10,
    "RANDOM_STATE": 42,
    "HIDDEN_FEATURES": 3072, 
    "NUM_BLOCKS": 2,
    "DROPOUT_RATE": 0.3832296399028748,     
    "LR": 1.7749224152998916e-05,             
    "WEIGHT_DECAY": 0.0008723639231133962,
    "MARGIN": 0.3898481154806984,
    "TEMPERATURE": 0.010879304099273002,
    "HYBRID_ALPHA": 0,
    "MIXUP_ALPHA": 0.23197266859653176,
    "LATENT_DIM": 2048,
    "KLD_WEIGHT": 1.9543552300752537e-06,
    "BATCH_SIZE": 256,
    "ACCUMULATION_STEPS": 2
}

TRAIN_DATA_PATH = "/kaggle/input/train.npz"
TEST_DATA_PATH = "/kaggle/input/test.clean.npz"
SUBMISSION_PATH = "submission_mlpvae.csv"

def main():
    seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    print(f"All seeds set to {SEED} and CUDNN is in deterministic mode.")
    
    all_text_embeddings, all_image_embeddings = train_fn(CONFIG, g, TRAIN_DATA_PATH)
    generate_submission(
        CONFIG,
        all_text_embeddings,
        all_image_embeddings,
        TEST_DATA_PATH
    )

if __name__ == "__main__":
    main()
