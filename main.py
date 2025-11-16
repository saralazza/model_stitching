import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import TextVariationalEncoder, TranslatorMLP, TripletLoss, PureContrastiveLoss
from utils import (
    seed_everything,
    seed_worker,
    load_training_data,
    calculate_mrr,
    mixup_data,
    generate_submission
)

SEED = 42

CONFIG = {
    "TEXT_VE_PATH_TPL": "best_text_ve_fold_{}.pth",
    "TRANSLATOR_PATH_TPL": "best_translator_mlp_fold_{}.pth",
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

def main():
    seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    print(f"All seeds set to {SEED} and CUDNN is in deterministic mode.")
    
    (
        all_text_embeddings,
        all_image_embeddings,
        num_captions_per_image,
        kfold,
        image_indices
    ) = load_training_data(CONFIG)
    
    all_fold_mrr = []
    
    for fold, (train_image_indices, val_image_indices) in enumerate(kfold.split(image_indices)):
        print(f"\n{'='*50}")
        print(f"--- STARTING FOLD {fold+1}/{CONFIG['N_FOLDS']} ---")
        print(f"{'='*50}")
    
        train_image_emb = all_image_embeddings[train_image_indices]
        val_image_emb = all_image_embeddings[val_image_indices]
    
        train_image_emb_expanded = train_image_emb.repeat_interleave(num_captions_per_image, dim=0)
    
        train_caption_indices = [i for idx in train_image_indices for i in range(idx * num_captions_per_image, (idx + 1) * num_captions_per_image)]
        val_caption_indices = [i for idx in val_image_indices for i in range(idx * num_captions_per_image, (idx + 1) * num_captions_per_image)]
    
        train_text_emb = all_text_embeddings[train_caption_indices]
        val_text_emb = all_text_embeddings[val_caption_indices]
        
        val_text_emb_one_per_image = val_text_emb[::num_captions_per_image]
    
        train_dataset = TensorDataset(train_text_emb, train_image_emb_expanded)
        val_dataset = TensorDataset(val_text_emb_one_per_image, val_image_emb)
    
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['BATCH_SIZE'], 
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'])
    
        print(f"Fold {fold+1}: Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
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
    
        criterion_triplet = TripletLoss(margin=CONFIG['MARGIN'])
        criterion_pure = PureContrastiveLoss(temperature=CONFIG['TEMPERATURE'])
    
        optimizer = optim.AdamW(
            itertools.chain(text_ve.parameters(), translator_model.parameters()),
            lr=CONFIG['LR'],
            weight_decay=CONFIG['WEIGHT_DECAY']
        )
        total_steps = len(train_loader) * CONFIG['EPOCHS']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        best_finetune_mrr = -1.0
        patience_counter = 0
    
        for epoch in range(CONFIG['EPOCHS']):
            text_ve.train()
            translator_model.train()
            total_train_loss = 0
            
            batch_iterator = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{CONFIG['EPOCHS']}")
            
            for i, (text_batch, image_batch) in enumerate(batch_iterator):
                text_batch = text_batch.to(CONFIG['DEVICE'])
                image_batch = image_batch.to(CONFIG['DEVICE'])
                
                target_embeddings = F.normalize(image_batch, p=2, dim=1)
                mixed_text, mixed_targets = mixup_data(text_batch, target_embeddings, alpha=CONFIG['MIXUP_ALPHA'])
    
                z, mu, log_var = text_ve(mixed_text)
                pred_embeddings = translator_model(z)
                
                loss_triplet = criterion_triplet(pred_embeddings, mixed_targets)
                loss_pure = criterion_pure(pred_embeddings, mixed_targets)
                hybrid_loss = (CONFIG['HYBRID_ALPHA'] * loss_triplet) + ((1 - CONFIG['HYBRID_ALPHA']) * loss_pure)
                
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                kld_loss = kld_loss / mixed_text.size(0)
                
                loss = hybrid_loss + (CONFIG['KLD_WEIGHT'] * kld_loss)
                loss = loss / CONFIG['ACCUMULATION_STEPS']
                loss.backward()
    
                if (i + 1) % CONFIG['ACCUMULATION_STEPS'] == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
    
                total_train_loss += loss.item() * CONFIG['ACCUMULATION_STEPS']
                batch_iterator.set_postfix({"Train Loss": f"{loss.item() * CONFIG['ACCUMULATION_STEPS']:.4f}"})
            
            avg_train_loss = total_train_loss / len(train_loader)
            val_mrr = calculate_mrr(text_ve, translator_model, val_loader, CONFIG['DEVICE'])
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"Fold {fold+1} Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Val MRR: {val_mrr:.4f} | LR: {current_lr:e}")
            
            if val_mrr > best_finetune_mrr:
                best_finetune_mrr = val_mrr
                
                text_ve_path = CONFIG['TEXT_VE_PATH_TPL'].format(fold)
                translator_path = CONFIG['TRANSLATOR_PATH_TPL'].format(fold)
                
                torch.save(text_ve.state_dict(), text_ve_path)
                torch.save(translator_model.state_dict(), translator_path)
                print(f"  🏆 New best MRR for Fold {fold+1}! Saving models to {text_ve_path}...")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['PATIENCE']:
                    print(f"Early stopping triggered for Fold {fold+1}.")
                    break
    
        print(f"\n✅ Fold {fold+1} finished. Best validation MRR: {best_finetune_mrr:.4f}")
        all_fold_mrr.append(best_finetune_mrr)
    
    print(f"\n{'='*50}")
    print(f"✅ All {CONFIG['N_FOLDS']} folds trained! ")
    print(f"Mean Validation MRR: {np.mean(all_fold_mrr):.4f} (Std: {np.std(all_fold_mrr):.4f})")
    print(f"{'='*50}")
    
    generate_submission(CONFIG, all_text_embeddings, all_image_embeddings)

if __name__ == "__main__":
    main()
