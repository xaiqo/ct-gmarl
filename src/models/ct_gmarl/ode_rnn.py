import torch
from torch import nn


class ODERNNCell(nn.Module):
    """
    Continuous-Time RNN cell using a Neural ODE for hidden state evolution.

    Equivalent to:
    h(t + delta_t) = h(t) + integral(f(h(tau), tau), t, t + delta_t)

    For efficiency in MARL rollouts, we use a fixed-step Euler or RK4 solver.
    """

    def __init__(
        self, input_size: int, hidden_size: int, solver: str = 'euler', steps: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.solver = solver
        self.steps = steps

        # The Derivative Function f_theta (the ODE right-hand side)
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, delta_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Advances the hidden state h by delta_t given input x.

        Args:
            x: Input tensor (batch, input_size)
            h: Hidden state (batch, hidden_size)
            delta_t: Normalized time interval (batch, 1)

        Returns:
            new_h: Updated hidden state
        """
        # We assume x is constant over the delta_t interval (Zero-Order Hold)
        dt_step = delta_t / self.steps

        curr_h = h
        for _ in range(self.steps):
            if self.solver == 'euler':
                # h_{n+1} = h_n + dt * f(h_n, x)
                derivative = self.ode_func(torch.cat([curr_h, x], dim=-1))
                curr_h = curr_h + dt_step * derivative
            elif self.solver == 'rk4':
                # Runge-Kutta 4th Order
                k1 = self.ode_func(torch.cat([curr_h, x], dim=-1))
                k2 = self.ode_func(torch.cat([curr_h + dt_step / 2 * k1, x], dim=-1))
                k3 = self.ode_func(torch.cat([curr_h + dt_step / 2 * k2, x], dim=-1))
                k4 = self.ode_func(torch.cat([curr_h + dt_step * k3, x], dim=-1))
                curr_h = curr_h + (dt_step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return curr_h


class CtGru(nn.Module):
    """
    Continuous-Time Gated Recurrent Unit.
    A more stable alternative to raw Neural ODEs for RL.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, delta_t: torch.Tensor
    ) -> torch.Tensor:
        # Standard GRU assumes delta_t = 1.0.
        # For CT-GRU, we scale the update by delta_t to simulate continuous decay.
        h_next = self.gru_cell(x, h)
        return (1.0 - delta_t) * h + delta_t * h_next
