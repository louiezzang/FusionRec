"""
FusionRec: Unified Contrastive-Attentive Fusion (Upgraded)
Reference implementation (PyTorch)

Upgrades in this file compared to the earlier template:
 - Dual-contrastive objectives: image↔text and text↔behavior with flexible lambda weighting
 - Automatic modality-aware pairing and safe handling when modalities are missing
 - Pretrained encoders placeholders (BERT, ResNet) clearly marked (keep HF & torchvision installed)
 - Batch / graph utilities hints for PyG-compatible usage
 - Training step that computes weighted loss: L = L_rec + lambda_it * L_img-text + lambda_tb * L_text-beh

Notes:
 - This file is a research/experiment template. Replace dataset/dataloader parts with your own pipeline.
 - Requires: transformers, torchvision, torch_geometric (optional if using graph encoder), torch

Author: Upgraded for dual-contrastive learning
"""

from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# (Optional) if you plan to use PyG graph layers, import them conditionally
try:
    from torch_geometric.nn import GraphSAGE, GATConv
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False

# (Optional) HuggingFace / torchvision imports (if available in your env)
try:
    from transformers import BertModel
except Exception:
    BertModel = None

try:
    from torchvision import models
except Exception:
    models = None

# ----------------------------- Utility ---------------------------------------
def _safe_zero(tensor: torch.Tensor, device: torch.device):
    return torch.tensor(0.0, device=device)

# ----------------------------- Encoders -------------------------------------
class ImageEncoder(nn.Module):
    """ResNet-50 based image encoder (requires torchvision)."""
    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        if models is None:
            raise RuntimeError("torchvision.models not available. Install torchvision or replace ImageEncoder.")
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.proj = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        with torch.no_grad():
            feats = self.backbone(images).view(images.size(0), -1)
        return self.proj(feats)


class TextEncoder(nn.Module):
    """BERT-based text encoder (requires transformers)."""
    def __init__(self, out_dim: int = 256, model_name: str = "bert-base-uncased", freeze_backbone: bool = False):
        super().__init__()
        if BertModel is None:
            raise RuntimeError("transformers.BertModel not available. Install transformers or replace TextEncoder.")
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.proj(pooled)


class BehaviorEncoder(nn.Module):
    """Item-side behavior/metadata encoder (MLP) or Node feature projection for graph encoder."""
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------- User Behavior Encoders --------------------------
class GraphUserEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 256, model_type: str = 'sage'):
        super().__init__()
        self.model_type = model_type
        if _HAS_PYG and model_type == 'gat':
            self.conv1 = GATConv(in_dim, hidden, heads=2, concat=False)
            self.conv2 = GATConv(hidden, out_dim, heads=2, concat=False)
        elif _HAS_PYG and model_type == 'sage':
            self.conv1 = GraphSAGE(in_dim, hidden)
            self.conv2 = GraphSAGE(hidden, out_dim)
        else:
            # fallback: simple MLP per-node (no graph propagation)
            self.conv1 = nn.Linear(in_dim, hidden)
            self.conv2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if _HAS_PYG and hasattr(self.conv1, '__call__') and ('GraphSAGE' in str(type(self.conv1)) or 'GATConv' in str(type(self.conv1))):
            # expect PyG-style conv: conv(x, edge_index)
            h = F.relu(self.conv1(x, edge_index))
            h = self.conv2(h, edge_index)
            return h
        else:
            h = F.relu(self.conv1(x))
            h = self.conv2(h)
            return h


class SeqUserEncoder(nn.Module):
    def __init__(self, item_emb_dim: int = 256, out_dim: int = 256, mode: str = 'transformer', max_len: int = 64):
        super().__init__()
        assert mode in ('transformer', 'gru')
        self.mode = mode
        self.item_proj = nn.Linear(item_emb_dim, out_dim)
        if mode == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=4, dim_feedforward=out_dim*2)
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.pos_emb = nn.Embedding(max_len, out_dim)
        else:
            self.gru = nn.GRU(input_size=out_dim, hidden_size=out_dim, batch_first=True)

    def forward(self, item_seq: torch.Tensor, seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = item_seq.shape
        h = self.item_proj(item_seq)
        if self.mode == 'transformer':
            positions = torch.arange(0, t, device=item_seq.device).unsqueeze(0).expand(b, -1)
            h = h + self.pos_emb(positions)
            h = h.transpose(0,1)
            key_padding_mask = None if seq_mask is None else (seq_mask == 0)
            out = self.transformer(h, src_key_padding_mask=key_padding_mask)
            out = out.transpose(0,1).mean(dim=1)
            return out
        else:
            out, _ = self.gru(h)
            return out.mean(dim=1)


# ----------------------------- Fusion Module ---------------------------------
class UserAwareAttentionFusion(nn.Module):
    def __init__(self, modality_dim: int = 256, user_dim: int = 512, hidden: int = 256):
        super().__init__()
        self.user_q = nn.Linear(user_dim, hidden)
        self.key_proj = nn.Linear(modality_dim, hidden)
        self.value_proj = nn.Linear(modality_dim, modality_dim)
        self.softmax = nn.Softmax(dim=1)
        self.out_proj = nn.Linear(modality_dim, modality_dim)

    def forward(self, modality_list: Dict[str, torch.Tensor], user_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        names = list(modality_list.keys())
        vals = torch.stack([modality_list[n] for n in names], dim=1)  # [B, M, D]
        keys = self.key_proj(vals)
        vals_v = self.value_proj(vals)
        q = self.user_q(user_vec).unsqueeze(2)  # [B, H, 1]
        logits = torch.matmul(keys, q).squeeze(2)  # [B, M]
        attn = self.softmax(logits)
        attn = attn.unsqueeze(2)
        fused = (vals_v * attn).sum(dim=1)
        out = self.out_proj(fused)
        # return also a map of modality->weight for analysis
        weights = {names[i]: attn[:, i, 0].detach() for i in range(len(names))}
        return out, weights


# ------------------------ Projection & Contrastive Loss -----------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.t()) / self.temperature
        diag_mask = torch.eye(2*batch_size, device=z.device).bool()
        sim.masked_fill_(diag_mask, -9e15)
        positives = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(z.device)
        labels = positives
        loss = F.cross_entropy(sim, labels)
        return loss


# ----------------------------- Recommendation Head ---------------------------
class RecommendationHead(nn.Module):
    def __init__(self, user_dim: int = 512, item_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(user_dim + item_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_vec, item_vec], dim=1)
        return self.mlp(x).squeeze(1)


# ------------------------------- FusionRec Model -----------------------------
class FusionRec(nn.Module):
    def __init__(self,
                 vocab_size: Optional[int] = None,
                 item_meta_dim: Optional[int] = None,
                 user_node_feat_dim: Optional[int] = None,
                 modality_dim: int = 256,
                 proj_dim: int = 128,
                 seq_mode: str = 'transformer',
                 text_model_name: str = 'bert-base-uncased'):
        super().__init__()
        # encoders (if transformers/torchvision not installed, user must replace with own classes)
        if BertModel is not None:
            self.text_enc = TextEncoder(out_dim=modality_dim, model_name=text_model_name)
        else:
            # fallback simple token embedding (user must adapt)
            self.text_enc = None

        if models is not None:
            self.image_enc = ImageEncoder(out_dim=modality_dim)
        else:
            self.image_enc = None

        self.behavior_enc = BehaviorEncoder(in_dim=item_meta_dim or 16, out_dim=modality_dim)

        self.graph_user = GraphUserEncoder(in_dim=user_node_feat_dim or modality_dim, out_dim=modality_dim)
        self.seq_user = SeqUserEncoder(item_emb_dim=modality_dim, out_dim=modality_dim, mode=seq_mode)

        # fusion and heads
        self.fusion = UserAwareAttentionFusion(modality_dim=modality_dim, user_dim=modality_dim*2)
        self.reco_head = RecommendationHead(user_dim=modality_dim*2, item_dim=modality_dim)

        # projection heads for contrastive alignment
        self.image_proj = ProjectionHead(modality_dim, proj_dim) if self.image_enc is not None else None
        self.text_proj = ProjectionHead(modality_dim, proj_dim) if self.text_enc is not None else None
        self.behavior_proj = ProjectionHead(modality_dim, proj_dim)

        # losses
        self.contrastive_loss = NTXentLoss(temperature=0.07)

    # --------------------- forward helpers ---------------------
    def forward_item_embeddings(self, image: Optional[torch.Tensor], text_ids: Optional[torch.Tensor], text_mask: Optional[torch.Tensor], item_meta: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        if (image is not None) and (self.image_enc is not None):
            out['image'] = self.image_enc(image)
        if (text_ids is not None) and (self.text_enc is not None):
            out['text'] = self.text_enc(text_ids, text_mask)
        if item_meta is not None:
            out['behavior'] = self.behavior_enc(item_meta)
        return out

    def forward_user_representations(self, user_node_feats: Optional[torch.Tensor], edge_index: Optional[torch.Tensor], seq_item_embs: Optional[torch.Tensor], seq_mask: Optional[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        if user_node_feats is not None:
            graph_vec = self.graph_user(user_node_feats, edge_index)
            # if graph_vec is per-node and we have a batch of users, assume the provided user_node_feats are per-batch
            # reduce to per-user vector (mean pooling) if needed
            if graph_vec.dim() == 3:
                graph_vec = graph_vec.mean(dim=1)
        else:
            graph_vec = torch.zeros(seq_item_embs.size(0), self.graph_user.conv2.out_features, device=device) if seq_item_embs is not None else torch.zeros(1, self.graph_user.conv2.out_features, device=device)

        if seq_item_embs is not None:
            seq_vec = self.seq_user(seq_item_embs, seq_mask)
        else:
            seq_vec = torch.zeros(graph_vec.size(0), graph_vec.size(1), device=device)

        return torch.cat([graph_vec, seq_vec], dim=1)

    def forward(self, batch: Dict) -> Dict:
        device = next(self.parameters()).device
        item_modalities = self.forward_item_embeddings(batch.get('image', None), batch.get('text_ids', None), batch.get('text_mask', None), batch.get('item_meta', None))
        user_vec = self.forward_user_representations(batch.get('user_node_feats', None), batch.get('edge_index', None), batch.get('seq_item_embs', None), batch.get('seq_mask', None))
        fused_item, attn_weights = self.fusion(item_modalities, user_vec)
        score = self.reco_head(user_vec, fused_item)

        proj = {}
        if 'image' in item_modalities and self.image_proj is not None:
            proj['image'] = self.image_proj(item_modalities['image'])
        if 'text' in item_modalities and self.text_proj is not None:
            proj['text'] = self.text_proj(item_modalities['text'])
        if 'behavior' in item_modalities and self.behavior_proj is not None:
            proj['behavior'] = self.behavior_proj(item_modalities['behavior'])

        return {'score': score, 'fused_item': fused_item, 'attn': attn_weights, 'modalities': item_modalities, 'projections': proj, 'user_vec': user_vec}


# ------------------------------- Training Step --------------------------------
def training_step(model: FusionRec,
                  batch: Dict,
                  optimizer: torch.optim.Optimizer,
                  recon_loss_fn = None,
                  lambda_img_text: float = 1.0,
                  lambda_text_beh: float = 1.0,
                  alpha_rec: float = 1.0) -> Dict:
    """
    Computes: loss = alpha_rec * L_rec + lambda_img_text * L_img_text + lambda_text_beh * L_text_behavior
    - L_rec : reconstruction / recommendation loss (BCE / BPR / pairwise)
    - L_img_text : contrastive between image and text (if both present in batch)
    - L_text_behavior : contrastive between text and behavior (if both present)

    The function automatically detects which projections exist in the forward outputs.
    """
    model.train()
    out = model(batch)
    score = out['score']
    device = score.device

    # Recommendation loss
    labels = batch.get('labels', None)
    if labels is None:
        loss_reco = torch.tensor(0.0, device=device)
    else:
        if recon_loss_fn is None:
            recon_loss_fn = nn.BCEWithLogitsLoss()
        loss_reco = recon_loss_fn(score, labels.float())

    # Contrastive losses (auto-detect available projections)
    proj = out['projections']
    loss_img_text = torch.tensor(0.0, device=device)
    loss_text_beh = torch.tensor(0.0, device=device)

    proj_keys = list(proj.keys())

    # Helper: safe get
    def _get_proj(key: str):
        return proj[key] if key in proj else None

    z_img = _get_proj('image')
    z_text = _get_proj('text')
    z_beh = _get_proj('behavior')

    # Compute image <-> text contrastive if both exist
    if (z_img is not None) and (z_text is not None):
        loss_img_text = model.contrastive_loss(z_img, z_text)

    # Compute text <-> behavior contrastive if both exist
    if (z_text is not None) and (z_beh is not None):
        loss_text_beh = model.contrastive_loss(z_text, z_beh)

    # Total loss
    loss = alpha_rec * loss_reco + lambda_img_text * loss_img_text + lambda_text_beh * loss_text_beh

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {'loss': loss.item(), 'loss_reco': loss_reco.item(), 'loss_img_text': loss_img_text.item(), 'loss_text_beh': loss_text_beh.item()}


# ------------------------------- Example usage --------------------------------
if __name__ == '__main__':
    # Quick smoke test (random tensors). Replace with real dataloader in practice.
    B = 4
    seq_len = 8
    vocab_size = 30522
    item_meta_dim = 16
    user_node_feat_dim = 32

    # Create model (if transformers/torchvision not installed, instantiate with caution)
    model = FusionRec(item_meta_dim=item_meta_dim, user_node_feat_dim=user_node_feat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Fake batch with all modalities
    batch = {
        'image': torch.randn(B, 3, 224, 224),
        'text_ids': torch.randint(0, 1000, (B, 32)),
        'text_mask': torch.ones(B, 32, dtype=torch.int64),
        'item_meta': torch.randn(B, item_meta_dim),
        'seq_item_embs': torch.randn(B, seq_len, 256),
        'seq_mask': torch.ones(B, seq_len, dtype=torch.int64),
        'user_node_feats': torch.randn(B, user_node_feat_dim),
        'edge_index': None,
        'labels': torch.randint(0,2,(B,)).float()
    }

    stats = training_step(model, batch, optimizer, lambda_img_text=1.0, lambda_text_beh=1.0)
    print(stats)
