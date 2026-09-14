"""Dueling Deep Q-Network (DuelingDQN) implementation for visual RL."""

from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture for Snake visual state inputs.

    Architecture:
        - Input shape: (B, 4, 64, 64)
        - Conv1: 4 -> 16 channels, kernel 8x8, stride 4
        - Conv2: 16 -> 32 channels, kernel 4x4, stride 2
        - Conv3: 32 -> 64 channels, kernel 3x3, stride 1
        - Flatten -> Shared Linear(flatten_dim, 256)
        - Value Stream V(s): Linear(256, 256) -> ReLU -> Linear(256, 1)
        - Advantage Stream A(s, a): Linear(256, 256) -> ReLU -> Linear(256, 4)
        - Aggregation: Q(s, a) = V(s) + (A(s, a) - mean_a(A(s, a)))
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_actions: int = 4,
        conv_channels: Sequence[int] = (16, 32, 64),
        fully_connected_units: int = 256,
        input_shape: Tuple[int, int] = (64, 64),
    ):
        """Initialize DuelingDQN network modules.

        Args:
            in_channels: Number of stacked input frames/channels (default: 4).
            num_actions: Number of discrete output actions (default: 4).
            conv_channels: Sequence of output channel counts for Conv2d layers (default: (16, 32, 64)).
            fully_connected_units: Number of units in shared FC layer and stream hidden layers (default: 256).
            input_shape: Spatial (H, W) dimensions of input observation (default: (64, 64)).
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_actions = num_actions
        self.conv_channels = list(conv_channels)
        self.fully_connected_units = fully_connected_units
        self.input_shape = input_shape

        # Conv feature extraction layers
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.conv_channels[0],
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.conv_channels[0],
            out_channels=self.conv_channels[1],
            kernel_size=4,
            stride=2,
        )
        self.conv3 = nn.Conv2d(
            in_channels=self.conv_channels[1],
            out_channels=self.conv_channels[2],
            kernel_size=3,
            stride=1,
        )

        # Compute flatten dimension automatically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            conv_out = self._forward_conv(dummy)
            self.flatten_dim = conv_out.view(1, -1).size(1)

        # Shared fully connected layer
        self.fc_shared = nn.Linear(self.flatten_dim, self.fully_connected_units)

        # Value stream: V(s) -> shape (B, 1)
        self.val_fc = nn.Linear(self.fully_connected_units, self.fully_connected_units)
        self.val_head = nn.Linear(self.fully_connected_units, 1)

        # Advantage stream: A(s, a) -> shape (B, num_actions)
        self.adv_fc = nn.Linear(self.fully_connected_units, self.fully_connected_units)
        self.adv_head = nn.Linear(self.fully_connected_units, self.num_actions)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through convolutional feature extraction layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward_value_advantage(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute unaggregated Value V(s) and Advantage A(s, a) streams.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Tuple of (V(s) tensor of shape (B, 1), A(s, a) tensor of shape (B, num_actions)).
        """
        conv_out = self._forward_conv(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        shared = F.relu(self.fc_shared(flattened))

        val_hidden = F.relu(self.val_fc(shared))
        v = self.val_head(val_hidden)

        adv_hidden = F.relu(self.adv_fc(shared))
        a = self.adv_head(adv_hidden)

        return v, a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-action Q-values Q(s, a) using dueling aggregation.

        Aggregation formula:
            Q(s, a) = V(s) + (A(s, a) - mean_a(A(s, a)))

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Q-values tensor of shape (B, num_actions).
        """
        v, a = self.forward_value_advantage(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
