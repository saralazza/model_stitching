import torch

from mlp_vae.model import train_fn as train_fn_mlp_vae
from mlp.model import train_fn as train_fn_mlp
from utils import seed_everything
from submission import generate_submission_mlp_vae, generate_submission_mlp

SEED = 42

CONFIG_MLP_VAE = {
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

CONFIG_MLP = {
    "MODEL_PATH_TPL": "best_mlp_fold_{}.pth",
    "SUBMISSION_PATH": "submission_mlp.csv",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "EPOCHS": 100,
    "PATIENCE": 25,
    "N_FOLDS": 10,
    "RANDOM_STATE": 42,
    "HIDDEN_FEATURES": 3072, 
    "NUM_BLOCKS": 2,
    "DROPOUT_RATE": 0.3832296399028748,     
    "LR": 1.7749224152998916e-05,             
    "WEIGHT_DECAY": 0.0008723639231133962,
    "MARGIN": 0.3898481154806984,
    "TEMPERATURE": 0.010879304099273002,
    "HYBRID_ALPHA": 0.0,
    "MIXUP_ALPHA": 0.23197266859653176,
    "LABEL_SMOOTHING": 0.12,
    "BATCH_SIZE": 256,
    "ACCUMULATION_STEPS": 2
}

TRAIN_DATA_PATH_VAE = "/kaggle/input/train.npz"
TRAIN_DATA_PATH_CLEAN = "/kaggle/input/aml-challenge-dataset/train.npz"
CLEAN_CAPTION_INDICES_PATH = "/kaggle/input/clean-dataset/clean_caption_indices.npy"
TEST_DATA_PATH = "/kaggle/input/test.clean.npz"

def main():
    seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    print(f"All seeds set to {SEED} and CUDNN is in deterministic mode.")
    
    all_text_embeddings_vae, all_image_embeddings_vae = train_fn_mlp_vae(CONFIG_MLP_VAE, g, TRAIN_DATA_PATH_VAE)
    generate_submission_mlp_vae(
        CONFIG_MLP_VAE,
        all_text_embeddings_vae,
        all_image_embeddings_vae,
        TEST_DATA_PATH,
        submission_path=CONFIG_MLP_VAE["SUBMISSION_PATH"]
    )

    all_text_embeddings_mlp, all_image_embeddings_mlp = train_fn_mlp(
        CONFIG_MLP,
        g,
        TRAIN_DATA_PATH_CLEAN,
        CLEAN_CAPTION_INDICES_PATH
    )
    generate_submission_mlp(
        CONFIG_MLP,
        all_text_embeddings_mlp,
        all_image_embeddings_mlp,
        TEST_DATA_PATH,
        submission_path=CONFIG_MLP["SUBMISSION_PATH"]
    )

if __name__ == "__main__":
    main()
