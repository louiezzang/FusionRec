FusionRec: Unified Contrastive-Attentive Fusion for Multi-modal and Behavior-aware Recommendation
===

## Introduction

This is the Pytorch implementation for our FusionRec paper:

>FusionRec: Unified Contrastive-Attentive Fusion for Multi-modal and Behavior-aware Recommendation

```
                             ┌────────────────────────┐
                             │   User Interaction     │
                             │     History (i₁ → i₂)  │
                             └───────┬────────────────┘
                                     │
                   ┌────────────────┴───────────────┐
                   ▼                                ▼
        ┌──────────────────────┐          ┌──────────────────────┐
        │  Behavior Graph      │          │ Sequence Encoder     │
        │  (GCN or GAT)        │          │ (Transformer or GRU) │
        └─────────┬────────────┘          └──────────┬───────────┘
                  │                                  │
              [Graph-based User Embed]        [Sequence-based User Embed]
                            └──────┬──────────────┬─────┘
                                   ▼              ▼
                              ┌──────────────┬─────────────┐
                              │  Modality Embeddings       │
                              │  (Text, Image, Behavior)   │
                              └──────────────┬─────────────┘
                                             ▼
                       ┌─────────────────────────────────────┐
                       │   User-conditioned Attention Fusion │
                       └─────────────────────────────────────┘
                                             ▼
                      ┌──────────────────────────────────────┐
                      │  Contrastive Loss + Recommendation   │
                      │     (CTR / BPR / Top-K Ranking)      │
                      └──────────────────────────────────────┘
```

```
[User Interaction Sequence]
         |
 ┌───────┴───────────────┐
 ▼                       ▼
[Behavior Graph]     [Sequence Encoder]
 (GCN / GAT)          (Transformer / GRU)
     |                      |
     └─────┬──────────┬─────┘
           ▼          ▼
     [Modality Embeddings]
   (Text, Image, Behavior)
           ▼
[User-conditioned Attention Fusion]
           ▼
[Contrastive Loss + Recommendation Head]
```


```
                [User ID]
                   │
       ┌───────────┴────────────┐
       ▼                        ▼
 [Behavior Graph]       [User Behavior Sequence]
 (User-Item GCN)         (Recent Clicks: i₁, i₂, ..., i_T)
       │                        │
user_graph_embed        user_seq_embed (Transformer)
       └───────────┬────────────┘
                   ▼
           Combined User Embed
                   ▼
        ▶ Attention Fusion / Contrastive / CTR
```

User Behavior Modeling Strategies
```
(a) GCN-only Behavior Embedding
┌──────────────┐
│ User-Item    │
│ Interaction  │
│ Graph        │
└─────┬────────┘
      ▼
┌──────────────┐
│ Graph Encoder│ (GCN/GAT)
└─────┬────────┘
      ▼
[User Graph Embed]
      ▼
[Downstream Tasks]

──────────────────────────────────────────────

(b) Transformer-only Behavior Encoding
┌────────────┐
│ User Seq:  │ (e.g., i₁ → i₂ → i₃)
│ [Click Logs]│
└─────┬───────┘
      ▼
┌─────────────────────┐
│ Transformer Encoder │ (Self-Attn)
└─────┬───────────────┘
      ▼
[User Seq Embed]
      ▼
[Downstream Tasks]

──────────────────────────────────────────────

(c) Hybrid: Graph-aware Seq Encoder
      (Graph → Transformer)
┌──────────────┐
│ User-Item    │
│ Interaction  │
│ Graph        │
└─────┬────────┘
      ▼
[Item Graph Embeddings] ←──────┐
                               │
       ┌────────────┐         │
       │ User Seq   │ (i₁→i₂) │
       └────┬───────┘         │
            ▼                 │
[Item Embeds via Graph]──────▶
            ▼
   Transformer Encoder
            ▼
      [User Seq Embed]
            ▼
     [Downstream Tasks]
```


Attention Fusion + Contrastive Head
```
                   ┌───────────────────────────┐
                   │   Modality Embeddings     │
                   │ Text / Image / Behavior   │
                   └─────────┬────┬────────────┘
                             │    │
                             ▼    ▼
                 [e_text]  [e_image] ... [e_behavior]

                             │
                             ▼
         ┌────────────────────────────────────────┐
         │  User-conditioned Attention Weights    │
         │  a_i = softmax(QᵤᵗKᵢ / √d)             │
         └────────────────────────────────────────┘
                             ▼
              [Weighted Sum: ∑ aᵢ · eᵢ = e_fused]

                             │
                             ▼
        ┌──────────────────────────────────────────┐
        │      e_fused  vs. item_embedding         │
        │     Contrastive Loss (InfoNCE or BPR)    │
        └──────────────────────────────────────────┘
```


## Environment Requirement
- python >= 3.9
- Pytorch >= 2.1.0


## Dataset

We provide three processed datasets: Baby, Sports, Clothing.

Download from Google Drive: [Baby/Sports/Clothing](https://drive.google.com/drive/folders/1tU4IxYbLXMkp_DbIOPGvCry16uPvolLk)

## Training
  ```
  cd ./src
  python main.py
  ```
## Performance Comparison
TBD

## Citation
TBD

## Acknowledgement
The structure of this code is  based on [MMRec](https://github.com/enoche/MMRec). Thank for their work.
