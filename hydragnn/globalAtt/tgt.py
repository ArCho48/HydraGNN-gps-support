import torch
import torch.nn as nn
import torch.nn.functional as F

class PreNormResidual(nn.Module):
    def __init__(self, dim, module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x + self.module(self.norm(x), *args, **kwargs)


class TripletGraphTransformerLayer(nn.Module):
    """
    One layer of the Triplet Graph Transformer (TGT-At) + Triplet Aggregation (TGT-Ag).
    """

    def __init__(
        self,
        dim_node: int,
        dim_edge: int,
        num_heads: int = 8,
        num_triplet_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        assert dim_node % num_heads == 0, "dim_node must be divisible by num_heads"
        assert dim_edge % num_triplet_heads == 0, "dim_edge must be divisible by num_triplet_heads"

        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.num_heads = num_heads
        self.num_triplet_heads = num_triplet_heads
        self.head_dim_node = dim_node // num_heads
        self.head_dim_edge = dim_edge // num_triplet_heads

        # -----------------------------------------------------------------------------
        #  (1) Edge-Augmented Node Attention (EGT) projections
        # -----------------------------------------------------------------------------
        self.q_proj = nn.Linear(dim_node, dim_node, bias=False)
        self.k_proj = nn.Linear(dim_node, dim_node, bias=False)
        self.v_proj = nn.Linear(dim_node, dim_node, bias=False)

        # From each e[i,j], produce 2 * num_heads scalars: (bias_pred, gate_pred)
        self.edge_to_bias_and_gate = nn.Linear(dim_edge, 2 * num_heads, bias=False)

        self.out_proj_node = nn.Linear(dim_node, dim_node, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        # -----------------------------------------------------------------------------
        #  (2) Node & Edge Feed-Forward Networks
        # -----------------------------------------------------------------------------
        self.ffn_node = nn.Sequential(
            nn.Linear(dim_node, dim_node * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_node * 4, dim_node),
            nn.Dropout(dropout)
        )
        self.node_ffn_block = PreNormResidual(dim_node, self.ffn_node)

        self.ffn_edge = nn.Sequential(
            nn.Linear(dim_edge, dim_edge * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_edge * 4, dim_edge),
            nn.Dropout(dropout)
        )
        self.edge_ffn_block = PreNormResidual(dim_edge, self.ffn_edge)

        # -----------------------------------------------------------------------------
        #  (3) EGT edge-update from node attention logits
        # -----------------------------------------------------------------------------
        self.attnlogit_to_edge = nn.Linear(num_heads, dim_edge, bias=True)

        # -----------------------------------------------------------------------------
        #  (4A) Full “Triplet Attention” (TGT-At) projections
        # -----------------------------------------------------------------------------
        self.proj_q_in = nn.Linear(dim_edge, num_triplet_heads * self.head_dim_edge, bias=False)
        self.proj_p_in = nn.Linear(dim_edge, num_triplet_heads * self.head_dim_edge, bias=False)
        self.proj_v_in = nn.Linear(dim_edge, num_triplet_heads * self.head_dim_edge, bias=False)

        self.proj_gate_and_bias = nn.Linear(dim_edge, 2 * num_triplet_heads, bias=False)
        self.triplet_out_proj = nn.Linear(2 * num_triplet_heads * self.head_dim_edge, dim_edge, bias=True)
        self.triplet_dropout = nn.Dropout(dropout)

        # -----------------------------------------------------------------------------
        #  (5) LayerNorms for Node & Edge channels
        # -----------------------------------------------------------------------------
        self.norm_node_attn = nn.LayerNorm(dim_node)
        self.norm_edge_attn = nn.LayerNorm(dim_edge)
        self.norm_triplet   = nn.LayerNorm(dim_edge)


    def forward(self, h: torch.Tensor, e: torch.Tensor):
        """
        h: [N, dim_node]
        e: [N, N, dim_edge]
        Returns:
          h_out: [N, dim_node]
          e_out: [N, N, dim_edge]
        """
        N = h.size(0)

        # -----------------------------------------------------------------------------
        #  STEP 1.  Edge-Augmented Node Attention (EGT)
        # -----------------------------------------------------------------------------
        q_all = self.q_proj(h).view(N, self.num_heads, self.head_dim_node)  # [N, H, d_h]
        k_all = self.k_proj(h).view(N, self.num_heads, self.head_dim_node)  # [N, H, d_h]
        v_all = self.v_proj(h).view(N, self.num_heads, self.head_dim_node)  # [N, H, d_h]

        qh = q_all.permute(1, 0, 2)  # [H, N, d_h]
        kh = k_all.permute(1, 0, 2)  # [H, N, d_h]
        scale = 1.0 / (self.head_dim_node ** 0.5)
        logits_base = torch.einsum('h i d, h j d -> h i j', qh, kh) * scale  # [H, N, N]

        bg = self.edge_to_bias_and_gate(e)                 # [N, N, 2H]
        bias = bg[..., :self.num_heads].permute(2, 0, 1)   # [H, N, N]
        gate = torch.sigmoid(bg[..., self.num_heads:].permute(2, 0, 1))  # [H, N, N]

        raw_logits = logits_base + bias                    # [H, N, N]
        gated_logits = raw_logits * gate                    # [H, N, N]
        attn_weights = torch.softmax(gated_logits, dim=-1) # [H, N, N]
        attn_weights = self.attn_dropout(attn_weights)

        vh = v_all.permute(1, 0, 2)  # [H, N, d_h]
        o_h = torch.einsum('h i j, h j d -> h i d', attn_weights, vh)  # [H, N, d_h]

        o_cat = o_h.permute(1, 0, 2).reshape(N, self.num_heads * self.head_dim_node)  # [N, dim_node]
        node_update = self.out_proj_node(o_cat)  # [N, dim_node]
        h = h + node_update                      # residual

        logits_for_edge = raw_logits.permute(1, 2, 0)          # [N, N, H]
        edge_from_node_attn = self.attnlogit_to_edge(logits_for_edge)  # [N, N, dim_edge]

        # -----------------------------------------------------------------------------
        #  STEP 2.  Edge-Channel Pre-Norm + FFN
        # -----------------------------------------------------------------------------
        e_norm = self.norm_edge_attn(e)               # [N, N, dim_edge]
        e_agg  = e_norm + edge_from_node_attn         # [N, N, dim_edge]
        e = e_agg + self.edge_ffn_block(e_agg)        # [N, N, dim_edge]

        # -----------------------------------------------------------------------------
        #  (A)  Full Triplet Attention (TGT-At)
        # -----------------------------------------------------------------------------
        e_norm_trip = self.norm_triplet(e)           # [N, N, dim_edge]
        new_edge_from_triplet = self._full_triplet_attention(e_norm_trip)
        e = e + new_edge_from_triplet                 # residual
        e = e + self.edge_ffn_block(e)               # final edge FFN

        # -----------------------------------------------------------------------------
        #  (B)  Triplet Aggregation (TGT-Ag)
        # -----------------------------------------------------------------------------
        # If you prefer the O(N^2) Triplet Aggregation variant, simply comment out
        # the “full_triplet_attention” lines above and uncomment the three lines below:
        #
        # e_norm_trip = self.norm_triplet(e)
        # new_edge_from_agg = self.triplet_aggregation(e_norm_trip)
        # e = e + new_edge_from_agg
        # e = e + self.edge_ffn_block(e)

        # -----------------------------------------------------------------------------
        #  STEP 3.  Node Channel FFN (Pre-Norm + FFN)
        # -----------------------------------------------------------------------------
        h = h + self.node_ffn_block(self.norm_node_attn(h))  # [N, dim_node]

        return h, e


    # -----------------------------------------------------------------------------
    #  (A)  Full Triplet Attention (TGT-At) helper (unchanged from earlier)
    # -----------------------------------------------------------------------------
    def _full_triplet_attention(self, e_norm_trip: torch.Tensor) -> torch.Tensor:
        N = e_norm_trip.size(0)
        Ht = self.num_triplet_heads
        dt = self.head_dim_edge

        q_in_all = self.proj_q_in(e_norm_trip) \
            .view(N, N, Ht, dt).permute(2, 0, 1, 3)  # [Ht, N, N, dt]
        p_in_all = self.proj_p_in(e_norm_trip) \
            .view(N, N, Ht, dt).permute(2, 0, 1, 3)  # [Ht, N, N, dt]
        v_in_all = self.proj_v_in(e_norm_trip) \
            .view(N, N, Ht, dt).permute(2, 0, 1, 3)  # [Ht, N, N, dt]

        gb = self.proj_gate_and_bias(e_norm_trip)  # [N, N, 2*Ht]
        gate_pre = gb[..., :Ht].permute(2, 0, 1)     # [Ht, N, N]
        bias_pre = gb[..., Ht:].permute(2, 0, 1)     # [Ht, N, N]
        gate_matrix = torch.sigmoid(gate_pre)
        bias_matrix = bias_pre

        scale_t = 1.0 / (dt ** 0.5)

        # -- Inward --
        dot_term = torch.einsum(
            'h i j d, h j k d -> h i j k', q_in_all, p_in_all
        ) * scale_t                                  # [Ht, N, N, N]
        bias_ik = bias_matrix.unsqueeze(2)           # [Ht, N, 1, N]
        raw_triplet_in = dot_term + bias_ik           # [Ht, N, N, N]
        gate_ik = gate_matrix.unsqueeze(2)            # [Ht, N, 1, N]
        gated_triplet_in = raw_triplet_in * gate_ik   # [Ht, N, N, N]
        attn_trip_in = torch.softmax(gated_triplet_in, dim=-1)  # [Ht, N, N, N]
        o_in = torch.einsum('h i j k, h j k d -> h i j d', attn_trip_in, v_in_all)  # [Ht, N, N, dt]

        # -- Outward --
        p_in_trans = p_in_all.permute(0, 2, 1, 3)      # [Ht, N, N, dt]
        v_in_trans = v_in_all.permute(0, 2, 1, 3)      # [Ht, N, N, dt]
        gate_trans = gate_matrix.permute(0, 2, 1)      # [Ht, N, N]
        bias_trans = bias_matrix.permute(0, 2, 1)      # [Ht, N, N]

        dot_term_out = torch.einsum(
            'h i j d, h k j d -> h i k j', q_in_all, p_in_trans
        ) * scale_t                                    # [Ht, N, N, N]
        bias_out = bias_trans.unsqueeze(2)             # [Ht, N, 1, N]
        raw_triplet_out = dot_term_out + bias_out      # [Ht, N, N, N]
        gate_out = gate_trans.unsqueeze(2)             # [Ht, N, 1, N]
        gated_triplet_out = raw_triplet_out * gate_out # [Ht, N, N, N]
        attn_trip_out = torch.softmax(gated_triplet_out, dim=2)  # [Ht, N, N, N]
        o_out = torch.einsum('h i k j, h k j d -> h i j d', attn_trip_out, v_in_trans)  # [Ht, N, N, dt]

        o_pair = torch.cat([o_in, o_out], dim=-1)  # [Ht, N, N, 2*dt]
        o_pair = o_pair.permute(1, 2, 0, 3).reshape(N, N, Ht * 2 * dt)  # [N, N, 2*Ht*dt]
        new_edge = self.triplet_out_proj(o_pair)  # [N, N, dim_edge]
        new_edge = self.triplet_dropout(new_edge)
        return new_edge


    # -----------------------------------------------------------------------------
    #  (B)  Triplet Aggregation (TGT-Ag) method
    # -----------------------------------------------------------------------------
    def triplet_aggregation(self, e_norm: torch.Tensor) -> torch.Tensor:
        """
        Implements the efficient “Triplet Aggregation” (TGT-Ag) ≈ O(N^2).
        Input:
          e_norm: [N, N, dim_edge]  (pre-norm’d edge embeddings)
        Output:
          new_edge_from_agg: [N, N, dim_edge]
        """
        N = e_norm.size(0)
        Ht = self.num_triplet_heads
        dt = self.head_dim_edge  # = dim_edge / Ht

        # 1) Project to v_in_all: [Ht, N, N, dt]
        v_proj = self.proj_v_in(e_norm)  # [N, N, Ht*dt]
        v_in_all = v_proj.view(N, N, Ht, dt).permute(2, 0, 1, 3)  # [Ht, N, N, dt]

        # 2) Project to gate & bias: [Ht, N, N]
        gb = self.proj_gate_and_bias(e_norm)  # [N, N, 2*Ht]
        gate_pre = gb[..., :Ht].permute(2, 0, 1)  # [Ht, N, N]
        bias_pre = gb[..., Ht:].permute(2, 0, 1)   # [Ht, N, N]
        gate_matrix = torch.sigmoid(gate_pre)  # [Ht, N, N]
        bias_matrix = bias_pre                 # [Ht, N, N]

        # 3) Inward aggregation weights: [Ht, N, N]
        weights_ik = torch.softmax(bias_matrix, dim=-1) * gate_matrix

        #    o_in[h,i,j,d] = Σ_k weights_ik[h,i,k] * v_in_all[h,j,k,d]
        o_in = torch.einsum('h i k, h j k d -> h i j d', weights_ik, v_in_all)  # [Ht, N, N, dt]

        # 4) Outward aggregation
        bias_trans = bias_matrix.permute(0, 2, 1)   # [Ht, N, N] (h,k,i)
        gate_trans = gate_matrix.permute(0, 2, 1)   # [Ht, N, N]
        weights_out = torch.softmax(bias_trans, dim=1) * gate_trans  # [Ht, N, N]

        v_out = v_in_all.permute(0, 2, 1, 3)  # [Ht, N, N, dt], dims [h, k, j, d]
        o_out = torch.einsum('h k i, h k j d -> h i j d', weights_out, v_out)  # [Ht, N, N, dt]

        # 5) Concat & project back to dim_edge
        o_pair = torch.cat([o_in, o_out], dim=-1)  # [Ht, N, N, 2*dt]
        o_pair = o_pair.permute(1, 2, 0, 3).reshape(N, N, Ht * 2 * dt)  # [N, N, 2*Ht*dt]
        new_edge_from_agg = self.triplet_out_proj(o_pair)  # [N, N, dim_edge]

        return new_edge_from_agg
