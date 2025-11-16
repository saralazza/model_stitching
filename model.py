import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TextVariationalEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.LayerNorm(hidden_features)
        )
        self.fc_mu = nn.Linear(hidden_features, latent_dim)
        self.fc_log_var = nn.Linear(hidden_features, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

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
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, pred_norm, target_norm):
        sim_matrix = torch.matmul(pred_norm, target_norm.T) / self.temperature
        labels = torch.arange(pred_norm.size(0), device=pred_norm.device)
        return F.cross_entropy(sim_matrix, labels)

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

def validation_fn(text_ve, translator_model, val_loader, criterion_triplet, criterion_pure, alpha, kld_weight, device):
    text_ve.eval()
    translator_model.eval()
    val_loss = 0
    with torch.no_grad():
        for text_batch, image_batch in val_loader:
            text_batch, image_batch = text_batch.to(device), image_batch.to(device)
            
            z, mu, log_var = text_ve(text_batch)
            pred_embeddings = translator_model(z)
            
            target_embeddings = F.normalize(image_batch, p=2, dim=1)
            
            loss_triplet = criterion_triplet(pred_embeddings, target_embeddings)
            loss_pure = criterion_pure(pred_embeddings, target_embeddings)
            hybrid_loss = (alpha * loss_triplet) + ((1 - alpha) * loss_pure)
            
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kld_loss = kld_loss / text_batch.size(0)
            
            loss = hybrid_loss + (kld_weight * kld_loss)
            val_loss += loss.item()
            
    return val_loss / len(val_loader)
