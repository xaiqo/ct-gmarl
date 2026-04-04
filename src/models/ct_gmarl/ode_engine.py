from typing import Tuple

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class ODEFunct(nn.Module):
    """
    Learned dynamics function f(h, t) for continuous-time hidden state evolution.

    This module defines the derivative of the hidden state with respect to
    time, effectively modeling the 'drift' of information between
    discrete SIEM alert observations.
    """

    def __init__(self, hidden_dim: int):
        super(ODEFunct, self).__init__()
        self.nfe = 0
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),  # Tanh ensures stable gradients for ODE integration
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Calculates the dh/dt for the current hidden state.

        Args:
            t (torch.Tensor): Current timestamp (scalar or batch).
            h (torch.Tensor): Current hidden state [Batch, Hidden].

        Returns:
            torch.Tensor: Derivative of the hidden state [Batch, Hidden].
        """
        self.nfe += 1
        return self.net(h)


class ContinuousTimeODE(nn.Module):
    """
    Advanced Neural ODE Engine using Adjoint Method for backpropagation.

    This engine evolves the hidden state h(t) to h(t+dt) by solving
    the initial value problem (IVP) defined by the ODEFunct.
    """

    def __init__(self, hidden_dim: int, solver: str = 'rk4'):
        super(ContinuousTimeODE, self).__init__()
        self.func = ODEFunct(hidden_dim)
        self.solver = solver
        self.hidden_dim = hidden_dim

    def forward(self, h0: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Initial Value Problem (IVP) Integration over Delta T.

        Args:
            h0 (torch.Tensor): Initial hidden state [Batch, Hidden].
            dt (torch.Tensor): Time interval(s) for integration [Batch, 1].

        Returns:
            Tuple[torch.Tensor, int]: Evolved hidden state h(t+dt) and NFE count.
        """
        # Note: torchdiffeq expects a t vector [0, dt].
        # For batch dt, we solve per sequence or pad.
        # Efficient approach: Normalize t to [0, 1] and scale the func by dt.

        # Wrapped func for batch dt scaling
        class ScaledODEFunc(nn.Module):
            def __init__(self, base_func, delta_t):
                super().__init__()
                self.base_func = base_func
                self.dt = delta_t

            def forward(self, t, h):
                return self.dt * self.base_func(t, h)

        # Integration points: [Start, End]
        t_span = torch.tensor([0.0, 1.0], device=h0.device)

        # Scaling the dynamics by dt allows batch integration of varying dt
        # dh/d(t_norm) = (dh/dt) * (dt/d(t_norm)) = f(h, t) * dt
        scaled_func = ScaledODEFunc(self.func, dt)

        # Solve IVP: h1 = h0 + integr(scaled_func) from 0 to 1
        h_seq = odeint(
            scaled_func, h0, t_span, method=self.solver, rtol=1e-3, atol=1e-4
        )

        nfe_count = self.func.nfe
        self.func.nfe = 0  # Reset for next call

        return h_seq[1], nfe_count  # Return the final state at t=1.0 and NFE


class ODERNNCell(nn.Module):
    """
    High-Fidelity ODE-RNN Cell for CT-GMARL.

    Combines GNN-processed spatial observations with Neural ODE-processed
    temporal dynamics to maintain a continuous, publication-grade POSMDP state.
    """

    def __init__(self, input_dim: int, hidden_dim: int, solver: str = 'rk4'):
        super(ODERNNCell, self).__init__()
        self.hidden_dim = hidden_dim

        # Spatial-to-Hidden projection (Observation update)
        self.input_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(),
        )
        self.update_layer = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim), nn.Tanh()
        )

        # Temporal engine (Continuous time evolution)
        self.ode_engine = ContinuousTimeODE(hidden_dim, solver=solver)

    def forward(
        self, obs_feat: torch.Tensor, h_prev: torch.Tensor, dt: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Single step of the ODE-RNN.

        Args:
            obs_feat (torch.Tensor): GNN features from current observation [Batch, Feat].
            h_prev (torch.Tensor): Previous hidden state [Batch, Hidden].
            dt (torch.Tensor): Time since last event [Batch, 1].

        Returns:
            Tuple[torch.Tensor, int]: New hidden state [Batch, Hidden] and NFE count.
        """
        # 1. Temporal Evolution (Hidden state 'drifts' over dt)
        h_evolved, nfe_count = self.ode_engine(h_prev, dt)

        # 2. Sequential Update (Discrete update from new observation)
        combined = torch.cat([obs_feat, h_evolved], dim=-1)
        gate = self.input_gate(combined)
        h_candidate = self.update_layer(combined)

        # Gating observation into evolved state
        h_new = (1 - gate) * h_evolved + gate * h_candidate

        return h_new, nfe_count
