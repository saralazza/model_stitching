import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..utils import calculate_mrr_mlp, seed_worker, mixup_data, load_data_cleaned

class GeGLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features * 2)
        self.gelu = nn.GELU()
    def forward(self, x):
        proj_out = self.proj(x)
        a, b = proj_out.chunk(2, dim=-1)
        return self.gelu(a) * b

class ResidualBlockGeGLU(nn.Module):
    def __init__(self, features, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            GeGLU(features, features), 
            nn.BatchNorm1d(features),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        return x + self.block(x)

class TranslatorMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_blocks, dropout_rate):
        super().__init__()
        
        backbone_layers = [
            nn.Linear(in_features, hidden_features, bias=False),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Dropout(0.5)
        ]
        for _ in range(num_blocks):
            backbone_layers.append(
                ResidualBlockGeGLU(hidden_features, dropout_rate)
            )
        self.backbone = nn.Sequential(*backbone_layers)
        
        self.translator_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.GELU(),
            nn.Linear(hidden_features // 2, out_features)
        )

    def forward(self, x):
        shared_representation = self.backbone(x)
        output = self.translator_head(shared_representation)
        return F.normalize(output, p=2, dim=1)

class PureContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, label_smoothing=0.12):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
    def forward(self, pred_norm, target_norm):
        sim_matrix = torch.matmul(pred_norm, target_norm.T) / self.temperature
        labels = torch.arange(pred_norm.size(0), device=pred_norm.device)
        return F.cross_entropy(sim_matrix, labels, label_smoothing=self.label_smoothing)

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    def forward(self, text_embeds, img_embeds_norm):
        sim_matrix = text_embeds @ img_embeds_norm.T
        positive_scores = torch.diag(sim_matrix)
        mask = torch.eye(text_embeds.size(0), dtype=torch.bool, device=text_embeds.device)
        sim_matrix_masked = sim_matrix.masked_fill(mask, -float('inf'))
        hard_negative_scores = sim_matrix_masked.max(dim=1)[0]
        return F.relu(self.margin - positive_scores + hard_negative_scores).mean()

def validation_fn(translator_model, val_loader, criterion_triplet, criterion_pure, alpha, device):
    translator_model.eval()
    val_loss = 0
    with torch.no_grad():
        for text_batch, image_batch in val_loader:
            text_batch, image_batch = text_batch.to(device), image_batch.to(device)
            
            pred_embeddings = translator_model(text_batch)
            target_embeddings = F.normalize(image_batch, p=2, dim=1)
            
            loss_triplet = criterion_triplet(pred_embeddings, target_embeddings)
            loss_pure = criterion_pure(pred_embeddings, target_embeddings)
            hybrid_loss = (alpha * loss_triplet) + ((1 - alpha) * loss_pure)
            
            val_loss += hybrid_loss.item()
            
    return val_loss / len(val_loader)

def train_fn(CONFIG, g, train_data_path, clean_caption_indices_path):
    (
        all_text_embeddings,
        all_image_embeddings,
        pair_indices,
        kfold
    ) = load_data_cleaned(
        train_data_path,
        clean_caption_indices_path,
        CONFIG['N_FOLDS'],
        CONFIG['RANDOM_STATE']
    )
    
    all_fold_mrr = []
    
    for fold, (train_pair_indices, val_pair_indices) in enumerate(kfold.split(pair_indices)):
        print(f"\n{'='*50}")
        print(f"--- STARTING FOLD {fold+1}/{CONFIG['N_FOLDS']} ---")
        print(f"{'='*50}")
    
        train_text_emb = all_text_embeddings[train_pair_indices]
        train_image_emb = all_image_embeddings[train_pair_indices]
        
        val_text_emb = all_text_embeddings[val_pair_indices]
        val_image_emb = all_image_embeddings[val_pair_indices]
    
        train_dataset = TensorDataset(train_text_emb, train_image_emb)
        val_dataset = TensorDataset(val_text_emb, val_image_emb)
    
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['BATCH_SIZE'], 
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'])
    
        print(f"Fold {fold+1}: Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
        translator_model = TranslatorMLP(
            in_features=all_text_embeddings.shape[1], 
            out_features=all_image_embeddings.shape[1], 
            hidden_features=CONFIG['HIDDEN_FEATURES'],
            num_blocks=CONFIG['NUM_BLOCKS'],
            dropout_rate=CONFIG['DROPOUT_RATE']
        ).to(CONFIG['DEVICE'])
    
        criterion_triplet = TripletLoss(margin=CONFIG['MARGIN'])
        criterion_pure = PureContrastiveLoss(
            temperature=CONFIG['TEMPERATURE'],
            label_smoothing=CONFIG['LABEL_SMOOTHING']
        )
    
        optimizer = optim.AdamW(
            translator_model.parameters(),
            lr=CONFIG['LR'],
            weight_decay=CONFIG['WEIGHT_DECAY']
        )
        
        total_steps = len(train_loader) * CONFIG['EPOCHS'] // CONFIG['ACCUMULATION_STEPS']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        best_finetune_mrr = -1.0
        patience_counter = 0
    
        for epoch in range(CONFIG['EPOCHS']):
            translator_model.train()
            total_train_loss = 0
            
            batch_iterator = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{CONFIG['EPOCHS']}")
            
            for i, (text_batch, image_batch) in enumerate(batch_iterator):
                text_batch = text_batch.to(CONFIG['DEVICE'])
                image_batch = image_batch.to(CONFIG['DEVICE'])
                
                target_embeddings = F.normalize(image_batch, p=2, dim=1)
                mixed_text, mixed_targets = mixup_data(text_batch, target_embeddings, alpha=CONFIG['MIXUP_ALPHA'])
    
                pred_embeddings = translator_model(mixed_text)
                
                loss_triplet = criterion_triplet(pred_embeddings, mixed_targets)
                loss_pure = criterion_pure(pred_embeddings, mixed_targets)
                loss = (CONFIG['HYBRID_ALPHA'] * loss_triplet) + ((1 - CONFIG['HYBRID_ALPHA']) * loss_pure)
                
                loss = loss / CONFIG['ACCUMULATION_STEPS']
                loss.backward()
    
                if (i + 1) % CONFIG['ACCUMULATION_STEPS'] == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
    
                total_train_loss += loss.item() * CONFIG['ACCUMULATION_STEPS']
                batch_iterator.set_postfix({"Train Loss": f"{loss.item() * CONFIG['ACCUMULATION_STEPS']:.4f}"})
            
            avg_train_loss = total_train_loss / len(train_loader)
            val_loss = validation_fn(
                translator_model, 
                val_loader, 
                criterion_triplet, 
                criterion_pure, 
                CONFIG['HYBRID_ALPHA'], 
                CONFIG['DEVICE']
            )
            val_mrr = calculate_mrr_mlp(translator_model, val_loader, CONFIG['DEVICE'])
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"Fold {fold+1} Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MRR: {val_mrr:.4f} | LR: {current_lr:e}")
            
            if val_mrr > best_finetune_mrr:
                best_finetune_mrr = val_mrr
                
                model_path = CONFIG['MODEL_PATH_TPL'].format(fold)
                torch.save(translator_model.state_dict(), model_path)
                print(f"  🏆 New best MRR for Fold {fold+1}! Saving model to {model_path}...")
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
    
    return all_text_embeddings, all_image_embeddings
