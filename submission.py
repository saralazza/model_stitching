import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .mlp_vae.model import TextVariationalEncoder, TranslatorMLP as TranslatorMLPVAE
from .mlp.model import TranslatorMLP as TranslatorMLPOnly

def generate_submission_mlp_vae(CONFIG, all_text_embeddings, all_image_embeddings, test_data_path):
    print(f"--- Generating {CONFIG['SUBMISSION_PATH']}") 
    
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
    
        translator_model = TranslatorMLPOnly(
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
    
        submission_df.to_csv(CONFIG['SUBMISSION_PATH'], index=False)
    
        print(f"\n✅ Ensembled submission file '{CONFIG['SUBMISSION_PATH']}' has been generated successfully!")
        print("Here's a preview of the first 5 rows:")
        print(submission_df.head())

def generate_submission_mlp(CONFIG, all_text_embeddings, all_image_embeddings, test_data_path, submission_path="submission.csv"):
    print("--- Generating submission.csv using K-Fold MLP models (Averaging) ---")
    
    try:
        test_data = np.load(test_data_path, allow_pickle=True)
        test_text_emb = torch.from_numpy(test_data['captions/embeddings']).float()
        test_caption_ids = test_data['captions/ids']
        test_loader = DataLoader(test_text_emb, batch_size=CONFIG['BATCH_SIZE'])
        print(f"Test data and caption IDs loaded: {len(test_caption_ids)} samples.")
    except FileNotFoundError:
        print("ERROR: Please make sure the 'test.clean.npz' file is in the correct directory.")
        return
    
    all_predictions_list = []
    
    for fold in range(CONFIG['N_FOLDS']):
        print(f"\n--- Generating predictions for Fold {fold+1}/{CONFIG['N_FOLDS']} ---")
        
        model_path = CONFIG['MODEL_PATH_TPL'].format(fold)
    
        translator_model = TranslatorMLPOnly(
            in_features=all_text_embeddings.shape[1],
            out_features=all_image_embeddings.shape[1],
            hidden_features=CONFIG['HIDDEN_FEATURES'],
            num_blocks=CONFIG['NUM_BLOCKS'],
            dropout_rate=CONFIG['DROPOUT_RATE']
        ).to(CONFIG['DEVICE'])
    
        try:
            translator_model.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print(f"ERROR: Could not find model file for fold {fold}: {model_path}")
            continue 
        
        translator_model.eval()
    
        fold_predictions = []
        with torch.no_grad():
            for text_batch in tqdm(test_loader, desc=f"Predicting Fold {fold+1}"):
                text_batch = text_batch.to(CONFIG['DEVICE'])
                
                pred_batch = translator_model(text_batch)
                
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
    
        submission_df.to_csv(CONFIG['SUBMISSION_PATH'], index=False)
    
        print(f"\n✅ Ensembled submission file '{CONFIG['SUBMISSION_PATH']}' has been generated successfully!")
        print("Here's a preview of the first 5 rows:")
        print(submission_df.head())
