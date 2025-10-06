
# https://chatgpt.com/g/g-p-68bc64a7ec1c81918a34c5bb0636fa28-research/c/689699c9-a1b0-832a-a32b-472bd37b3b43

"""
FusionRec: Unified Contrastive-Attentive Fusion (PyTorch) - Integrated GraphUserEncoder Update

This file is an integrated FusionRec implementation with:
 - PyG-compatible GraphUserEncoder supporting GraphSAGE / GAT / MLP-fallback
 - Dual contrastive objectives (image↔text, text↔behavior) with automatic modality detection
 - Pretrained encoder support (BERT, ResNet) with safe fallbacks
 - User-conditioned attention fusion and recommendation head
 - Robust training_step with lambda-weighted loss composition

Notes:
 - Requires optional packages for full functionality: transformers, torchvision, torch_geometric
 - This file replaces the previous Fusionrec Model and includes the updated GraphUserEncoder

Author: Updated for user request
"""

from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional imports
try:
    from torch_geometric.nn import GraphSAGE, GAT
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False

try:
    from transformers import BertModel
except Exception:
    BertModel = None

try:
    from torchvision import models
except Exception:
    models = None

# ----------------------------- Utility ---------------------------------------
def _safe_zero(device: torch.device):
    return torch.tensor(0.0, device=device)

# ----------------------------- Encoders -------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        if models is None:
            # simple MLP fallback
            self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3*224*224, out_dim))
            self.proj = nn.Identity()
            self._fallback = True
        else:
            resnet = models.resnet50(pretrained=pretrained)
            modules = list(resnet.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.proj = nn.Linear(resnet.fc.in_features, out_dim)
            self._fallback = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self._fallback:
            # assume images flattened
            return self.proj(self.backbone(images))
        with torch.no_grad():
            feats = self.backbone(images).view(images.size(0), -1)
        return self.proj(feats)


class TextEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, model_name: str = "bert-base-uncased", freeze_backbone: bool = False):
        super().__init__()
        if BertModel is None:
            # fallback: token embedding + pooling
            self._fallback = True
            self.token_emb = nn.Embedding(30522, out_dim)
            self.pool = lambda x: x.mean(dim=1)
        else:
            self._fallback = False
            self.bert = BertModel.from_pretrained(model_name)
            if freeze_backbone:
                for p in self.bert.parameters():
                    p.requires_grad = False
            self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._fallback:
            emb = self.token_emb(input_ids)
            pooled = emb.mean(dim=1)
            return pooled
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        return self.proj(pooled)


class BehaviorEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------- GraphUserEncoder (updated) ----------------------
class GraphUserEncoder(nn.Module):
    """
    Unified Graph-based User Encoder
    - Supports GraphSAGE and GAT when PyG is available (PyG 2.x style)
    - Falls back to an MLP if PyG not installed
    """
    def __init__(self, in_dim: int, out_dim: int, model_type: str = 'sage', hidden_dim: Optional[int] = None, num_layers: int = 2, heads: int = 2):
        super().__init__()
        self.model_type = model_type.lower()
        hidden_dim = hidden_dim or out_dim

        if not _HAS_PYG:
            # fallback MLP
            self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
            self._use_pyg = False
            return

        self._use_pyg = True
        if self.model_type == 'sage':
            # GraphSAGE(in_channels, hidden_channels, num_layers, out_channels=None, ...)
            self.encoder = GraphSAGE(in_channels=in_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=out_dim)
        elif self.model_type == 'gat':
            # GAT(in_channels, hidden_channels, num_layers, out_channels=None, heads=1, ...)
            self.encoder = GAT(in_channels=in_dim, hidden_channels=hidden_dim, num_layers=num_layers, out_channels=out_dim, heads=heads)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self._use_pyg:
            return self.encoder(x)
        # PyG convs expect (x, edge_index)
        return self.encoder(x, edge_index)


# -------------------------- Sequence User Encoder ---------------------------
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
                 text_model_name: str = 'bert-base-uncased',
                 graph_type: str = 'sage',
                 graph_num_layers: int = 2):
        super().__init__()
        # encoders
        self.text_enc = TextEncoder(out_dim=modality_dim, model_name=text_model_name) if BertModel is not None else None
        self.image_enc = ImageEncoder(out_dim=modality_dim) if models is not None else None
        self.behavior_enc = BehaviorEncoder(in_dim=item_meta_dim or 16, out_dim=modality_dim)

        # user encoders
        self.graph_user = GraphUserEncoder(in_dim=user_node_feat_dim or modality_dim, out_dim=modality_dim, model_type=graph_type, num_layers=graph_num_layers)
        self.seq_user = SeqUserEncoder(item_emb_dim=modality_dim, out_dim=modality_dim, mode=seq_mode)

        # fusion and heads
        self.fusion = UserAwareAttentionFusion(modality_dim=modality_dim, user_dim=modality_dim*2)
        self.reco_head = RecommendationHead(user_dim=modality_dim*2, item_dim=modality_dim)

        # projection heads
        self.image_proj = ProjectionHead(modality_dim, proj_dim) if self.image_enc is not None else None
        self.text_proj = ProjectionHead(modality_dim, proj_dim) if self.text_enc is not None else None
        self.behavior_proj = ProjectionHead(modality_dim, proj_dim)

        # loss
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
        # graph vector
        if user_node_feats is not None:
            graph_vec = self.graph_user(user_node_feats, edge_index)  # expected shape: [B, D] or [N, D]
            # if graph returns node-level, reduce to per-batch user vector when needed (user provides per-batch features)
            if graph_vec.dim() == 3:
                graph_vec = graph_vec.mean(dim=1)
        else:
            # fallback zeros
            graph_vec = torch.zeros(seq_item_embs.size(0), self.graph_user.encoder[-1].out_features if not _HAS_PYG else self.graph_user.encoder.out_channels, device=device) if seq_item_embs is not None else torch.zeros(1, self.behavior_enc.net[-1].out_features, device=device)

        # seq vector
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
    model.train()
    out = model(batch)
    score = out['score']
    device = score.device

    # Recommendation loss
    labels = batch.get('labels', None)
    if labels is None:
        loss_reco = _safe_zero(device)
    else:
        if recon_loss_fn is None:
            recon_loss_fn = nn.BCEWithLogitsLoss()
        loss_reco = recon_loss_fn(score, labels.float())

    # Contrastive losses
    proj = out['projections']
    loss_img_text = _safe_zero(device)
    loss_text_beh = _safe_zero(device)

    z_img = proj.get('image', None)
    z_text = proj.get('text', None)
    z_beh = proj.get('behavior', None)

    if (z_img is not None) and (z_text is not None):
        loss_img_text = model.contrastive_loss(z_img, z_text)

    if (z_text is not None) and (z_beh is not None):
        loss_text_beh = model.contrastive_loss(z_text, z_beh)

    loss = alpha_rec * loss_reco + lambda_img_text * loss_img_text + lambda_text_beh * loss_text_beh

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {'loss': loss.item(), 'loss_reco': loss_reco.item() if isinstance(loss_reco, torch.Tensor) else float(loss_reco), 'loss_img_text': loss_img_text.item() if isinstance(loss_img_text, torch.Tensor) else float(loss_img_text), 'loss_text_beh': loss_text_beh.item() if isinstance(loss_text_beh, torch.Tensor) else float(loss_text_beh)}


# ------------------------------- Example usage --------------------------------
if __name__ == '__main__':
    B = 4
    seq_len = 8
    item_meta_dim = 16
    user_node_feat_dim = 32

    model = FusionRec(item_meta_dim=item_meta_dim, user_node_feat_dim=user_node_feat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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




######### PyG 기반 예제 DataLoader 코드
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np

def build_toy_graph(num_users=3, num_items=5, user_feat_dim=8, item_feat_dim=8):
    """
    Creates a toy user-item interaction graph with:
      - Bipartite edges (user -> item)
      - Simple random feature vectors for both users & items
    """
    total_nodes = num_users + num_items
    user_nodes = torch.arange(0, num_users)
    item_nodes = torch.arange(num_users, num_users + num_items)

    # sample edges: each user interacts with random items
    src, dst = [], []
    for u in user_nodes:
        interacted_items = np.random.choice(item_nodes, size=2, replace=False)
        for i in interacted_items:
            src.append(u)
            dst.append(i)

    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)  # undirected

    # node features (user + item)
    user_feats = torch.randn(num_users, user_feat_dim)
    item_feats = torch.randn(num_items, item_feat_dim)
    x = torch.cat([user_feats, item_feats], dim=0)

    data = Data(x=x, edge_index=edge_index)
    data.user_nodes = user_nodes
    data.item_nodes = item_nodes
    return data


def build_toy_dataloader(batch_size=2):
    """
    Wrap toy graphs into a small PyG DataLoader
    """
    dataset = [build_toy_graph() for _ in range(5)]  # create 5 small graphs
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader



######### 간단한 모델 호출 예시
from fusionrec_model import FusionRec  # assuming FusionRec is defined
import torch

# Create toy loader
loader = build_toy_dataloader()

# Instantiate model
model = FusionRec(
    text_model="bert-base-uncased",
    image_model="resnet18",
    graph_model_type="sage",  # or "gat"
)

# Mock inputs
text_emb = torch.randn(4, 256)
image_emb = torch.randn(4, 256)

for batch in loader:
    x, edge_index = batch.x, batch.edge_index
    print(f"x.shape={x.shape}, edge_index.shape={edge_index.shape}")

    # graph user embedding
    user_emb = model.graph_user(x, edge_index)
    print("Graph user embedding:", user_emb.shape)
    break
