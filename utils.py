import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Seeder function for DataLoader workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_training_data(CONFIG, train_data_path):
    try:
        full_train_data = np.load(train_data_path, allow_pickle=True)
        all_text_embeddings = torch.from_numpy(full_train_data['captions/embeddings']).float()
        all_image_embeddings = torch.from_numpy(full_train_data['images/embeddings']).float()

        num_images = all_image_embeddings.shape[0]
        num_captions_per_image = all_text_embeddings.shape[0] // num_images

        image_indices = np.arange(num_images)
        kfold = KFold(n_splits=CONFIG['N_FOLDS'], shuffle=True, random_state=CONFIG['RANDOM_STATE'])

        print(f"Data loaded: {all_text_embeddings.shape[0]} text embeddings, {all_image_embeddings.shape[0]} image embeddings.")
        print(f"KFold splitter created with {CONFIG['N_FOLDS']} splits.")

        return all_text_embeddings, all_image_embeddings, num_captions_per_image, kfold, image_indices
    except FileNotFoundError:
        print("Please make sure the training data file is in the correct directory.")
        raise

def calculate_mrr_mlp_vae(text_ve, translator_model, val_loader, device):
    text_ve.eval()
    translator_model.eval()
    all_reciprocal_ranks = []
    with torch.no_grad():
        for text_batch, image_batch in val_loader:
            text_batch = text_batch.to(device)
            image_batch = image_batch.to(device)
            
            z, _, _ = text_ve(text_batch)
            pred_embeddings_norm = translator_model(z)
            
            target_embeddings_norm = F.normalize(image_batch, p=2, dim=1)
            sim_matrix = pred_embeddings_norm @ target_embeddings_norm.T
            
            correct_scores = torch.diag(sim_matrix).unsqueeze(1)
            ranks = (sim_matrix >= correct_scores).sum(dim=1).float()
            reciprocal_ranks = 1.0 / ranks
            all_reciprocal_ranks.append(reciprocal_ranks)
            
    mrr = torch.cat(all_reciprocal_ranks).mean().item()
    return mrr

def calculate_mrr_mlp(translator_model, val_loader, device):
    translator_model.eval()
    all_reciprocal_ranks = []
    with torch.no_grad():
        for text_batch, image_batch in val_loader:
            text_batch = text_batch.to(device)
            image_batch = image_batch.to(device)
            
            pred_embeddings_norm = translator_model(text_batch)
            target_embeddings_norm = F.normalize(image_batch, p=2, dim=1)
            sim_matrix = pred_embeddings_norm @ target_embeddings_norm.T
            
            correct_scores = torch.diag(sim_matrix).unsqueeze(1)
            ranks = (sim_matrix >= correct_scores).sum(dim=1).float()
            reciprocal_ranks = 1.0 / ranks
            all_reciprocal_ranks.append(reciprocal_ranks)
            
    mrr = torch.cat(all_reciprocal_ranks).mean().item()
    return mrr

def mixup_data(x, y, alpha=0.2):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size(0)
    indices = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[indices, :]
    mixed_y = lam * y + (1 - lam) * y[indices, :]
    return mixed_x, mixed_y

def load_data_cleaned(train_data_path, clean_caption_indices_path, n_folds, random_state):
    try:
        full_train_data = np.load(train_data_path, allow_pickle=True)
        all_text_embeddings_full = torch.from_numpy(full_train_data['captions/embeddings']).float()
        all_image_embeddings_full = torch.from_numpy(full_train_data['images/embeddings']).float()
        all_image_ids_full = full_train_data['images/names']

        num_images_total = all_image_embeddings_full.shape[0]
        num_captions_per_image = all_text_embeddings_full.shape[0] // num_images_total

        print(f"Loading clean indices from '{clean_caption_indices_path}'...")
        clean_caption_indices = np.load(clean_caption_indices_path)
        print(f"Loaded {len(clean_caption_indices)} clean caption indices.")

        print("Filtering embeddings based on clean indices...")
        all_text_embeddings = all_text_embeddings_full[clean_caption_indices]
        corresponding_image_indices = clean_caption_indices // num_captions_per_image
        all_image_embeddings = all_image_embeddings_full[corresponding_image_indices]

        print(f"Clean data shape (text): {all_text_embeddings.shape}")
        print(f"Clean data shape (image): {all_image_embeddings.shape}")

        num_clean_pairs = len(all_text_embeddings)
        pair_indices = np.arange(num_clean_pairs)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        print(f"\nCLEAN Data loaded: {num_clean_pairs} text/image pairs.")
        print(f"KFold splitter created for {num_clean_pairs} pairs.")

        return all_text_embeddings, all_image_embeddings, pair_indices, kfold
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

def generate_final_submission(csv_path_mlp, csv_path_vae_mlp, output_path, weight_mlp=0.65, weight_vae_mlp=0.35):
    df_mlp = pd.read_csv(csv_path_mlp)
    df_vae_mlp = pd.read_csv(csv_path_vae_mlp)

    embs_mlp = np.array([json.loads(e) for e in df_mlp['embedding']])
    embs_vae_mlp = np.array([json.loads(e) for e in df_vae_mlp['embedding']])

    avg_embs = (weight_mlp * embs_mlp) + (weight_vae_mlp * embs_vae_mlp)
    embedding_json_list = [json.dumps(embedding.tolist()) for embedding in avg_embs]

    df_ensembled = pd.DataFrame({
        'id': df_mlp['id'],
        'embedding': embedding_json_list
    })

    df_ensembled.to_csv(output_path, index=False)
    print(f"Ensembled submission created at '{output_path}'")
    print(df_ensembled.head())