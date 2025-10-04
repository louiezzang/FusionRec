"""
FusionRec: Unified Contrastive-Attentive Fusion
Reference implementation (PyTorch)

This is a lightweight, readable template for the FusionRec architecture discussed earlier.
It implements:
 - ImageEncoder (simple CNN backbone placeholder)
 - TextEncoder (token embedding + transformer pooling placeholder)
 - BehaviorEncoder (MLP over metadata)
 - GraphUserEncoder (dense GCN-style layer)
 - SeqUserEncoder (Transformer/GRU option)
 - UserAwareAttentionFusion (user-conditioned attention over modalities)
 - Projection heads and Contrastive (NT-Xent) loss
 - Recommendation scoring head (dot product / MLP)

Notes:
 - Replace placeholder encoders with pretrained models (e.g., ResNet, BERT) for production.
 - Graph encoder here uses dense adjacency multiplication for clarity; swap for PyG or DGL for scalability.
 - This file is intended as a starting point for experiments, ablations and integration.

Author: Generated for user
"""

from typing import Optional, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Basic Encoders ---------------------------------
class ImageEncoder(nn.Module):
    """Simple CNN-based image encoder placeholder. Replace with ResNet/timm for production."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)


class TextEncoder(nn.Module):
    """Simple token embedding + pooling. Replace with pretrained Transformer encoder for better results."""
    def __init__(self, vocab_size: int, emb_dim: int = 256, out_dim: int = 256, max_len: int = 128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, dim_feedforward=emb_dim*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.project = nn.Linear(emb_dim, out_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # token_ids: [B, T]
        b, t = token_ids.shape
        positions = torch.arange(0, t, device=token_ids.device).unsqueeze(0).expand(b, -1)
        h = self.token_emb(token_ids) + self.pos_emb(positions)
        # transformer expects [T, B, D]
        h = h.transpose(0, 1)
        if attention_mask is not None:
            # PyTorch Transformer uses src_key_padding_mask with shape [B, T]
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        h = h.mean(dim=0)  # [B, D]
        return self.project(h)


class BehaviorEncoder(nn.Module):
    """Encodes item-side behavior / metadata features (numerical or categorical pre-embedded)."""
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# -------------------------- User Behavior Encoders ----------------------------
class GraphUserEncoder(nn.Module):
    """
    Dense GCN-style graph encoder for user nodes.
    For clarity we accept a dense adjacency (or normalized adjacency) A: [N, N] and node features X: [N, D]
    In practice use sparse libraries (PyG/DGL) for large graphs.
    """
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 256, num_layers: int = 2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * (num_layers-1) + [out_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        # X: [N, D]; A: [N, N] (should be normalized)
        h = X
        if A is not None:
            # one round of aggregation
            h = torch.matmul(A, h)
        return self.net(h)


class SeqUserEncoder(nn.Module):
    """Sequence encoder: choose Transformer or GRU style through argument."""
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
        # item_seq: [B, T, item_emb_dim]
        b, t, _ = item_seq.shape
        h = self.item_proj(item_seq)  # [B, T, out_dim]
        if self.mode == 'transformer':
            positions = torch.arange(0, t, device=item_seq.device).unsqueeze(0).expand(b, -1)
            h = h + self.pos_emb(positions)
            h = h.transpose(0,1)  # [T, B, D]
            key_padding_mask = None
            if seq_mask is not None:
                key_padding_mask = (seq_mask == 0)
            out = self.transformer(h, src_key_padding_mask=key_padding_mask)  # [T, B, D]
            out = out.transpose(0,1).mean(dim=1)
            return out
        else:
            out, _ = self.gru(h)
            return out.mean(dim=1)


# ----------------------------- Fusion Module ---------------------------------
class UserAwareAttentionFusion(nn.Module):
    """
    Fuse modality vectors (list of tensors [B, D]) into a single item representation
    conditioned on a user vector (or concatenation of dual-view user vectors).
    """
    def __init__(self, modality_dim: int = 256, user_dim: int = 256, hidden: int = 256):
        super().__init__()
        # project user to query; modalities to keys/values
        self.user_q = nn.Linear(user_dim, hidden)
        self.key_proj = nn.Linear(modality_dim, hidden)
        self.value_proj = nn.Linear(modality_dim, modality_dim)
        self.softmax = nn.Softmax(dim=1)
        self.out_proj = nn.Linear(modality_dim, modality_dim)

    def forward(self, modality_list: Dict[str, torch.Tensor], user_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # modality_list: dict of name->[B, D]
        # user_vec: [B, user_dim]
        names = list(modality_list.keys())
        vals = torch.stack([modality_list[n] for n in names], dim=1)  # [B, M, D]
        keys = self.key_proj(vals)  # [B, M, H]
        vals_v = self.value_proj(vals)  # [B, M, D]
        q = self.user_q(user_vec).unsqueeze(2)  # [B, H, 1]
        # compute attention logits: q^T k_i
        logits = torch.matmul(keys, q).squeeze(2)  # [B, M]
        attn = self.softmax(logits)
        attn = attn.unsqueeze(2)  # [B, M, 1]
        fused = (vals_v * attn).sum(dim=1)  # [B, D]
        out = self.out_proj(fused)
        return out, attn.squeeze(2)  # return fused vector and attention weights


# ------------------------ Projection & Contrastive Loss -----------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss (SimCLR style)
    Assumes inputs z_i and z_j are representations of two views for the same batch.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1, z2: [B, D]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.t()) / self.temperature  # [2B,2B]
        # mask out self-similarities
        diag_mask = torch.eye(2*batch_size, device=z.device).bool()
        sim.masked_fill_(diag_mask, -9e15)
        # positive pairs: (i, i+B) and (i+B, i)
        positives = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(z.device)
        labels = positives
        loss = F.cross_entropy(sim, labels)
        return loss


# ----------------------------- Recommendation Head ---------------------------
class RecommendationHead(nn.Module):
    """Simple score function between user vector and item vector."""
    def __init__(self, user_dim: int = 256, item_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(user_dim + item_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user_vec, item_vec], dim=1)
        return self.mlp(x).squeeze(1)


# ------------------------------- FusionRec Model -----------------------------
class FusionRec(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 item_meta_dim: int,
                 user_node_feat_dim: int,
                 modality_dim: int = 256,
                 proj_dim: int = 128,
                 seq_mode: str = 'transformer'):
        super().__init__()
        # encoders
        self.image_enc = ImageEncoder(out_dim=modality_dim)
        self.text_enc = TextEncoder(vocab_size=vocab_size, emb_dim=modality_dim, out_dim=modality_dim)
        self.behavior_enc = BehaviorEncoder(in_dim=item_meta_dim, out_dim=modality_dim)

        # user encoders
        self.graph_user = GraphUserEncoder(in_dim=user_node_feat_dim, out_dim=modality_dim)
        self.seq_user = SeqUserEncoder(item_emb_dim=modality_dim, out_dim=modality_dim, mode=seq_mode)

        # fusion and heads
        self.fusion = UserAwareAttentionFusion(modality_dim=modality_dim, user_dim=modality_dim*2)
        self.reco_head = RecommendationHead(user_dim=modality_dim*2, item_dim=modality_dim)

        # projection heads for contrastive alignment
        self.image_proj = ProjectionHead(modality_dim, proj_dim)
        self.text_proj = ProjectionHead(modality_dim, proj_dim)
        self.behavior_proj = ProjectionHead(modality_dim, proj_dim)

        # losses
        self.contrastive_loss = NTXentLoss(temperature=0.07)

    def forward_item_embeddings(self, image: Optional[torch.Tensor], text_ids: Optional[torch.Tensor], text_mask: Optional[torch.Tensor], item_meta: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # returns modality embeddings dict for items (B, D)
        out = {}
        if image is not None:
            out['image'] = self.image_enc(image)
        if text_ids is not None:
            out['text'] = self.text_enc(text_ids, text_mask)
        if item_meta is not None:
            out['behavior'] = self.behavior_enc(item_meta)
        return out

    def forward_user_representations(self, user_node_feats: Optional[torch.Tensor], A: Optional[torch.Tensor], seq_item_embs: Optional[torch.Tensor], seq_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # returns concatenated user vector [B, 2D]
        # Graph user: user_node_feats [N, Dn] and adjacency A [N, N] -> we assume batch corresponds to subset; for per-batch use adapt accordingly.
        # seq_user: item sequence embeddings [B, T, D]
        # For simplicity, assume user_node_feats and seq_item_embs are provided per-batch and aligned with mini-batch users.
        graph_vec = self.graph_user(user_node_feats, A) if user_node_feats is not None else torch.zeros(user_node_feats.size(0), self.graph_user.net[-1].out_features, device=next(self.parameters()).device)
        # if graph_vec returns [N, D] and we are using a batch of users, pick matching rows; here assume graph_vec is already per-batch
        seq_vec = self.seq_user(seq_item_embs, seq_mask) if seq_item_embs is not None else torch.zeros(graph_vec.size(0), graph_vec.size(1), device=next(self.parameters()).device)
        return torch.cat([graph_vec, seq_vec], dim=1)

    def forward(self, batch: Dict) -> Dict:
        """
        batch keys (examples):
         - image: [B, 3, H, W]
         - text_ids: [B, T]
         - text_mask: [B, T]
         - item_meta: [B, M]
         - seq_item_embs: [B, T, D] (precomputed item embeddings for sequence)
         - seq_mask: [B, T]
         - user_node_feats: [B, Dn]
         - adj: [B, B] or None
         - user_ids / item_ids - optional
         - labels: target scores or clicks
        """
        device = next(self.parameters()).device
        # 1) item modality embeddings
        item_modalities = self.forward_item_embeddings(batch.get('image', None), batch.get('text_ids', None), batch.get('text_mask', None), batch.get('item_meta', None))

        # 2) user representation (dual-view)
        user_vec = self.forward_user_representations(batch.get('user_node_feats', None), batch.get('adj', None), batch.get('seq_item_embs', None), batch.get('seq_mask', None))

        # 3) fuse using user conditioned attention
        fused_item, attn = self.fusion(item_modalities, user_vec)

        # 4) recommendation score
        score = self.reco_head(user_vec, fused_item)

        # 5) contrastive projections (use available pairs - here do pairwise between modalities present)
        proj = {}
        for k in item_modalities:
            if k == 'image': proj['image'] = self.image_proj(item_modalities[k])
            if k == 'text': proj['text'] = self.text_proj(item_modalities[k])
            if k == 'behavior': proj['behavior'] = self.behavior_proj(item_modalities[k])

        out = {'score': score, 'fused_item': fused_item, 'attn': attn, 'modalities': item_modalities, 'projections': proj, 'user_vec': user_vec}
        return out


# ------------------------------- Training Step --------------------------------
def training_step(model: FusionRec, batch: Dict, optimizer: torch.optim.Optimizer, recon_loss_fn = None, alpha_contrastive: float = 1.0) -> Dict:
    model.train()
    out = model(batch)
    score = out['score']
    labels = batch.get('labels', None)
    loss_reco = torch.tensor(0.0, device=score.device)
    if labels is not None:
        if recon_loss_fn is None:
            # default: BCE with logits for implicit feedback
            recon_loss_fn = nn.BCEWithLogitsLoss()
        loss_reco = recon_loss_fn(score, labels.float())

    # contrastive: if at least two projections available
    proj = out['projections']
    loss_contrastive = torch.tensor(0.0, device=score.device)
    proj_keys = list(proj.keys())
    if len(proj_keys) >= 2:
        # choose a simple pairing: image <-> text if exists, else first two
        if 'image' in proj and 'text' in proj:
            loss_contrastive = model.contrastive_loss(proj['image'], proj['text'])
        else:
            loss_contrastive = model.contrastive_loss(proj[proj_keys[0]], proj[proj_keys[1]])

    loss = loss_reco + alpha_contrastive * loss_contrastive
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {'loss': loss.item(), 'loss_reco': loss_reco.item(), 'loss_contrastive': loss_contrastive.item()}


# ------------------------------- Usage Example --------------------------------
if __name__ == '__main__':
    # quick smoke test with random tensors
    B = 8
    vocab_size = 5000
    item_meta_dim = 16
    user_node_feat_dim = 32

    model = FusionRec(vocab_size=vocab_size, item_meta_dim=item_meta_dim, user_node_feat_dim=user_node_feat_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # fake batch
    batch = {
        'image': torch.randn(B, 3, 64, 64),
        'text_ids': torch.randint(1, vocab_size, (B, 32)),
        'text_mask': torch.ones(B, 32, dtype=torch.int64),
        'item_meta': torch.randn(B, item_meta_dim),
        'seq_item_embs': torch.randn(B, 10, 256),
        'seq_mask': torch.ones(B, 10, dtype=torch.int64),
        'user_node_feats': torch.randn(B, user_node_feat_dim),
        'adj': None,
        'labels': torch.randint(0,2,(B,))
    }

    stats = training_step(model, batch, optimizer)
    print(stats)
