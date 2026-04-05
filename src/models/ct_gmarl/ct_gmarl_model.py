from typing import List, Tuple

import torch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from .gat_processor import MultiHeadGAT
from .ode_engine import ODERNNCell as TrueODERNNCell


class CtGmarlModel(TorchRNN, nn.Module):
    """
    Continuous-Time Graph MARL Model.

    Integrates Graph Attention Networks (GAT) for topology reasoning and
    Neural ODEs for processing irregularly sampled (asynchronous) SIEM alerts.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.cell_size = model_config.get('custom_model_config', {}).get(
            'lstm_cell_size', 128
        )

        # 1. SIEM & State Feature Extractors
        # Inputs: obs (256) + siem_embedding (128) = 384
        self.fc_obs = nn.Linear(256 + 128, 256)

        # 2. Graph Attention layer for spatial reasoning (e.g. node relationships)
        # NeurIPS Grade: MultiHeadGAT creates informational bottlenecks through topological masking
        self.gat = MultiHeadGAT(in_features=256, n_hidden=128, n_heads=4)

        # 3. Neural ODE Cell for temporal dynamics using Adjoint method (torchdiffeq)
        self.ode_rnn = TrueODERNNCell(
            input_dim=128, hidden_dim=self.cell_size, solver='rk4'
        )

        # 5. Output Branches
        self.action_branch = nn.Linear(self.cell_size, num_outputs)
        self.value_branch = nn.Linear(self.cell_size, 1)

        self._cur_value = None

    @override(TorchRNN)
    def forward_rnn(
        self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Forward pass handling the flattened Gymnasium Dict space.

        Indices (Alphabetical concatenation by RLlib):
        - action_mask [0:62]
        - delta_t [62:63]
        - obs [63:319]
        - siem_embedding [319:447]
        """
        # indices for flattened array: action_mask [0:132], adj_matrix [132:10132], delta_t [10132:10133], obs [10133:10389], siem_embedding [10389:10517]
        action_mask = inputs[:, :, :132]
        adj_matrix = inputs[:, :, 132:10132].view(-1, inputs.shape[1], 100, 100)
        delta_t_norm = inputs[:, :, 10132:10133]
        obs = inputs[:, :, 10133:10389]
        siem_emb = inputs[:, :, 10389:10517]

        # 1. Fuse Raw Obs + NLP Telemetry
        x = torch.cat([obs, siem_emb], dim=-1)
        x = torch.relu(self.fc_obs(x)) # Shape: (Batch, Seq, 256)

        # 2. Apply Graph Attention (Batch, Seq, Features)
        batch_size, seq_len, _ = x.size()
        
        # We now use the actual adjacency matrix from the environment!
        # The mask generation from gat_processor maps 0 connectivity to -1e9
        from .gat_processor import SubnetMaskGenerator
        adj_raw = adj_matrix.view(batch_size * seq_len, 100, 100)
        adj = SubnetMaskGenerator.create_mask(100, adj_raw).to(x.device)

        # Proper Node Formulation for Graph Attention
        # Instead of repeating the identical flat observation, we treat Node 0 as the 
        # Gateway/Super node that receives the full telemetry, allowing the MultiHeadGAT 
        # to reason over the topology and pass the information logically.
        x_flat = x.view(batch_size * seq_len, 256)
        x_nodes = torch.zeros(batch_size * seq_len, 100, 256, device=x.device)
        x_nodes[:, 0, :] = x_flat
        
        x_gat = self.gat(x_nodes, adj) # Shape (Batch*Seq, 100, 128)
        
        # Pool node embeddings back to graph embedding
        x_gat = torch.mean(x_gat, dim=1).view(batch_size, seq_len, 128)

        # 3. Continuous-Time Neural ODE Integration
        # We iterate over the sequence if seq_len > 1 (during training)
        h_t = state[0]
        outputs = []

        for t in range(seq_len):
            xt = x_gat[:, t, :]
            dt = delta_t_norm[:, t, :]
            h_t, nfe = self.ode_rnn(xt, h_t, dt)
            outputs.append(h_t)

        x_rnn = torch.stack(outputs, dim=1)

        # 4. Finalize Output
        logits = self.action_branch(x_rnn)
        self._cur_value = torch.reshape(self.value_branch(x_rnn), [-1])

        # Apply Action Mask
        masked_logits = torch.where(
            action_mask == 0.0,
            torch.tensor(-1e10, device=logits.device, dtype=logits.dtype),
            logits,
        )

        return masked_logits, [h_t]

    @override(TorchRNN)
    def value_function(self) -> TensorType:
        return self._cur_value

    @override(TorchRNN)
    def get_initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(self.cell_size, dtype=torch.float32),
        ]
