from typing import List, Tuple

import torch
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from .graph_attention import GATLayer, TopologyMessagePasser
from .ode_rnn import ODERNNCell


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
        self.gat = GATLayer(in_features=256, out_features=128)

        # 3. Neural ODE Cell for temporal dynamics
        self.ode_rnn = ODERNNCell(
            input_size=128, hidden_size=self.cell_size, solver='rk4'
        )

        # 4. Topology-based Message Passing (DMZ -> Internal -> Restricted)
        self.message_passer = TopologyMessagePasser(self.cell_size)

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
        action_mask = inputs[:, :, :62]
        delta_t_norm = inputs[:, :, 62:63]
        obs = inputs[:, :, 63:319]
        siem_emb = inputs[:, :, 319:447]

        # 1. Fuse Raw Obs + NLP Telemetry
        x = torch.cat([obs, siem_emb], dim=-1)
        x = torch.relu(self.fc_obs(x))

        # 2. Apply Graph Attention (Batch, Seq, Features)
        # Note: In a real simulation edge case, we'd pass a real adjacency matrix.
        # For the model's self-loop baseline, we use identity.
        batch_size, seq_len, _ = x.size()
        adj = torch.eye(1).to(x.device).repeat(batch_size * seq_len, 1, 1)

        # Reshape for GAT (Batch*Seq, 1, Features) since GAT handles one graph at a time
        x_gat = x.view(-1, 1, 256)
        x_gat = self.gat(x_gat, adj).view(batch_size, seq_len, 128)

        # 3. Continuous-Time Neural ODE Integration
        # We iterate over the sequence if seq_len > 1 (during training)
        h_t = state[0]
        outputs = []

        for t in range(seq_len):
            xt = x_gat[:, t, :]
            dt = delta_t_norm[:, t, :]
            h_t = self.ode_rnn(xt, h_t, dt)
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
