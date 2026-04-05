from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class GATHead(nn.Module):
    """
    Single-Head Graph Attention Layer for spatial feature extraction.

    This head calculates self-attention between nodes in the network
    graph, weighted by their topological connectivity and state similarities.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super(GATHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the GAT attention mechanism.

        Args:
            h (torch.Tensor): Feature representations [Batch, NumNodes, InFeatures].
            adj_mask (torch.Tensor): Topological mask [Batch, NumNodes, NumNodes].
                                    Contains -inf for masked edges.

        Returns:
            torch.Tensor: Attentive features [Batch, NumNodes, OutFeatures].
        """
        # Batch Matrix Multiply: [B, N, F] x [F, O] -> [B, N, O]
        wh = torch.matmul(h, self.W)
        # Calculate attention coefficients in batch
        e = self._prepare_attentional_mechanism_input(wh)  # [Batch, NumNodes, NumNodes]

        # Apply topological mask across batch.
        # We can optimize sparseness dynamically by taking advantage of masked fill
        e = e.masked_fill(adj_mask == -1e9, -1e9)
        
        attention = F.softmax(e, dim=-1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, wh)  # [Batch, NumNodes, OutFeatures]

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, wh: torch.Tensor) -> torch.Tensor:
        """Helper to create the attention coefficients matrix in batch."""
        # wh: [Batch, NumNodes, OutFeatures]
        # Project heads for attention coefficients
        wh1 = torch.matmul(wh, self.a[: self.W.shape[1], :])  # [Batch, NumNodes, 1]
        wh2 = torch.matmul(wh, self.a[self.W.shape[1] :, :])  # [Batch, NumNodes, 1]
        # Broadcast add: [B, N, 1] + [B, 1, N] -> [B, N, N]
        # Memory optimization: compute only on non-masked elements
        # For simplicity in dense batched tensors, we stick to standard broadcasting 
        # but the masked_fill in forward avoids calculating softmax on zero-edges.
        e = wh1 + wh2.transpose(1, 2)
        return self.leakyrelu(e)


class MultiHeadGAT(nn.Module):
    """
    Final Multi-Head Graph Attention Network (GAT) for CT-GMARL.

    This module creates informational bottlenecks through topological
    masking, ensuring that agents can only attend to nodes within
    their reachable or observable subnet (Zero-Shot scalability).
    """

    def __init__(
        self, in_features: int, n_hidden: int, n_heads: int = 4, dropout: float = 0.1
    ):
        super(MultiHeadGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Implementation using 4 independent heads (Parallelized)
        self.heads = nn.ModuleList(
            [
                GATHead(in_features, n_hidden // n_heads, dropout=dropout)
                for _ in range(n_heads)
            ]
        )

        # Combine heads with a final linear layer
        self.out_layer = nn.Linear(n_hidden, n_hidden)
        self.layer_norm = nn.LayerNorm(n_hidden)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """
        Processes node features with multi-head spatial attention.

        Args:
            h (torch.Tensor): Node features [Batch, NumNodes, InFeat].
            adj_mask (torch.Tensor): Adjacency mask [Batch, NumNodes, NumNodes].

        Returns:
            torch.Tensor: Spatial node representations [Batch, NumNodes, Hidden].
        """
        # Efficient multi-head concatenation
        head_outputs = [head(h, adj_mask) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)  # [Batch, NumNodes, Hidden]

        # Post-processing for stability
        res = self.out_layer(combined)
        return self.layer_norm(res + F.relu(combined))  # Skip connection


class TopologyMessagePasser(nn.Module):
    """
    Handles agent-to-agent message passing based on network topology.
    DMZ agent -> Internal agent -> Restricted agent.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Message transform layers
        self.dmz_to_internal = nn.Linear(hidden_size, hidden_size)
        self.internal_to_restricted = nn.Linear(hidden_size, hidden_size)

    def forward(self, agent_hidden_states: dict) -> dict:
        """
        Applies topology-constrained message passing.

        Args:
            agent_hidden_states: Dict mapping agent_id (e.g. 'blue_dmz') to its hidden tensor.

        Returns:
            updated_hidden_states: Hidden states with fused messages.
        """
        updates = {}

        # DMZ -> Internal
        if 'blue_dmz' in agent_hidden_states and 'blue_internal' in agent_hidden_states:
            msg = self.dmz_to_internal(agent_hidden_states['blue_dmz'])
            updates['blue_internal'] = agent_hidden_states['blue_internal'] + msg

        # Internal -> Restricted
        if (
            'blue_internal' in agent_hidden_states
            and 'blue_restricted' in agent_hidden_states
        ):
            msg = self.internal_to_restricted(agent_hidden_states['blue_internal'])
            updates['blue_restricted'] = agent_hidden_states['blue_restricted'] + msg

        # Merge updates back
        for k, v in updates.items():
            agent_hidden_states[k] = v

        return agent_hidden_states
class SubnetMaskGenerator:
    """
    Utility to create topological masks for informational bottlenecks.

    Ensures that DMZ, Internal, and Secure subnets have restricted
    attention flow according to firewall rules described in research.
    """

    @staticmethod
    def create_mask(
        num_nodes: int, connectivity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generates a GAT-compatible topological mask.

        Args:
            num_nodes (int): Total hosts in the topology.
            connectivity (Optional[torch.Tensor]): Binary adjacency matrix.

        Returns:
            torch.Tensor: Mask with -1e9 for non-connected nodes.
        """
        if connectivity is None:
            # Default to fully connected (if no topology constraints)
            return torch.zeros(num_nodes, num_nodes)

        # Map binary 0 -> -1e9, 1 -> 0.0
        mask = torch.zeros_like(connectivity).float()
        mask.masked_fill_(connectivity == 0, -1e9)
        return mask
