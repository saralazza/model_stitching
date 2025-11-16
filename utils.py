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

from model import TextVariationalEncoder, TranslatorMLP

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

def calculate_mrr(text_ve, translator_model, val_loader, device):
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

def mixup_data(x, y, alpha=0.2):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size(0)
    indices = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[indices, :]
    mixed_y = lam * y + (1 - lam) * y[indices, :]
    return mixed_x, mixed_y

def generate_submission(CONFIG, all_text_embeddings, all_image_embeddings, test_data_path):
    print("--- Generating submission.csv") 
    
    try:
        test_data = np.load(test_data_path, allow_pickle=True)
        test_text_emb = torch.from_numpy(test_data['captions/embeddings']).float()
        test_caption_ids = test_data['captions/ids']
        test_loader = DataLoader(test_text_emb, batch_size=CONFIG['BATCH_SIZE'])
        print(f"Test data and caption IDs loaded: {len(test_caption_ids)} samples.")
    except FileNotFoundError:
        print("ERROR: Please make sure the test data file is in the correct directory.")
        return
    
    all_predictions_list = []
    
    for fold in range(CONFIG['N_FOLDS']):
        print(f"--- Generating predictions for Fold {fold+1}/{CONFIG['N_FOLDS']} ---")
        
        text_ve_path = CONFIG['TEXT_VE_PATH_TPL'].format(fold)
        translator_path = CONFIG['TRANSLATOR_PATH_TPL'].format(fold)
    
        text_ve = TextVariationalEncoder(
            in_features=all_text_embeddings.shape[1],
            hidden_features=CONFIG['LATENT_DIM'],
            latent_dim=CONFIG['LATENT_DIM']
        ).to(CONFIG['DEVICE'])
    
        translator_model = TranslatorMLP(
            in_features=CONFIG['LATENT_DIM'],
            out_features=all_image_embeddings.shape[1], 
            hidden_features=CONFIG['HIDDEN_FEATURES'],
            num_blocks=CONFIG['NUM_BLOCKS'],
            dropout_rate=CONFIG['DROPOUT_RATE']
        ).to(CONFIG['DEVICE'])
    
        try:
            text_ve.load_state_dict(torch.load(text_ve_path))
            translator_model.load_state_dict(torch.load(translator_path))
        except FileNotFoundError:
            print(f"ERROR: Could not find model files for fold {fold}: {text_ve_path} or {translator_path}")
            print("Please ensure training is complete and files are saved correctly.")
            continue 
        
        text_ve.eval()
        translator_model.eval()
    
        fold_predictions = []
        with torch.no_grad():
            for text_batch in tqdm(test_loader, desc=f"Predicting Fold {fold+1}"):
                text_batch = text_batch.to(CONFIG['DEVICE'])
                
                z, _, _ = text_ve(text_batch)
                pred_batch = translator_model(z)
                
                fold_predictions.append(pred_batch.cpu().numpy())
        
        all_predictions_list.append(np.concatenate(fold_predictions, axis=0))
    
    if not all_predictions_list:
        print("\nNo predictions were generated. Cannot create submission file.")
    else:
        print(f"\n--- Averaging predictions from {len(all_predictions_list)} folds ---")
        avg_predictions = np.mean(all_predictions_list, axis=0)
        
        embedding_json_list = [json.dumps(embedding.tolist()) for embedding in avg_predictions]
    
        submission_df = pd.DataFrame({
            'id': test_caption_ids,
            'embedding': embedding_json_list
        })
    
        submission_df.to_csv("submission.csv", index=False)
    
        print(f"\n✅ Ensembled submission file 'submission.csv' has been generated successfully!")
        print("Here's a preview of the first 5 rows:")
        print(submission_df.head())
