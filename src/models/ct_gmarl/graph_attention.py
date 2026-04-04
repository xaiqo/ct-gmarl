import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class GATLayer(nn.Module):
    """
    Simple Graph Attention Network (GAT) layer.
    Computes node embeddings by attending to neighbors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)
        """
        wh = torch.matmul(h, self.W)  # (batch, num_nodes, out_features)

        # Attention mechanism
        batch_size, num_nodes, _ = wh.size()

        # Self-attention score calculation
        # a_input: (batch, num_nodes, num_nodes, 2*out_features)
        a_input = torch.cat(
            [
                wh.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, -1),
                wh.repeat(1, num_nodes, 1),
            ],
            dim=-1,
        ).view(batch_size, num_nodes, num_nodes, 2 * self.out_features)

        e = self.leakyrelu(
            torch.matmul(a_input, self.a).squeeze(3)
        )  # (batch, num_nodes, num_nodes)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, wh)  # (batch, num_nodes, out_features)
        return h_prime


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
