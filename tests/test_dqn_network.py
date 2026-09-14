"""Unit tests for DuelingDQN network architecture."""

import pytest
import torch

from src.agents.dqn.network import DuelingDQN
from src.common.config import load_config


class TestDuelingDQNNetwork:
    """Test suite for DuelingDQN network behavior, shape compliance, and gradients."""

    def test_instantiation_default_and_config(self):
        """1. Verify network can be instantiated with default parameters and from config."""
        net = DuelingDQN()
        assert isinstance(net, torch.nn.Module)

        # Load snake_dqn.yaml config and instantiate matching parameters
        config = load_config("configs/snake_dqn.yaml")
        net_from_config = DuelingDQN(
            in_channels=config["state"]["shape"][0],
            num_actions=4,
            conv_channels=config["network"]["conv_channels"],
            fully_connected_units=config["network"]["fully_connected_units"],
            input_shape=(config["state"]["shape"][1], config["state"]["shape"][2]),
        )
        assert isinstance(net_from_config, torch.nn.Module)

    def test_batch_output_shape(self):
        """2. Verify a tensor with shape (2, 4, 64, 64) produces output shape (2, 4)."""
        net = DuelingDQN()
        x = torch.randn(2, 4, 64, 64)
        out = net(x)
        assert out.shape == (2, 4)

    def test_single_observation_batch_dimension(self):
        """3. Verify a single observation with batch dimension (1, 4, 64, 64) works."""
        net = DuelingDQN()
        x = torch.randn(1, 4, 64, 64)
        out = net(x)
        assert out.shape == (1, 4)

    def test_output_finite_values(self):
        """4. Verify output contains finite values (no NaN or Inf)."""
        net = DuelingDQN()
        torch.manual_seed(42)
        x = torch.randn(4, 4, 64, 64)
        out = net(x)
        assert torch.isfinite(out).all().item()

    def test_dueling_aggregation_numerical_correctness(self):
        """5. Verify Value and Advantage streams are combined using Q = V + A - mean(A)."""
        net = DuelingDQN()
        torch.manual_seed(100)
        x = torch.randn(3, 4, 64, 64)

        # Obtain raw streams
        v, a = net.forward_value_advantage(x)
        assert v.shape == (3, 1)
        assert a.shape == (3, 4)

        # Expected Q using dueling formula
        expected_q = v + (a - a.mean(dim=1, keepdim=True))

        # Forward Q
        actual_q = net(x)

        # Numerical comparison
        assert torch.allclose(actual_q, expected_q, atol=1e-6)

    def test_convolution_channel_progression(self):
        """6. Verify the network has the expected convolution channel progression: 4 -> 16 -> 32 -> 64."""
        net = DuelingDQN()

        assert net.conv1.in_channels == 4
        assert net.conv1.out_channels == 16

        assert net.conv2.in_channels == 16
        assert net.conv2.out_channels == 32

        assert net.conv3.in_channels == 32
        assert net.conv3.out_channels == 64

    def test_shared_fc_units(self):
        """7. Verify final shared fully connected representation has 256 units."""
        net = DuelingDQN()
        assert net.fc_shared.out_features == 256

    def test_cpu_execution(self):
        """8. Verify the network supports CPU execution explicitly."""
        net = DuelingDQN().to("cpu")
        x = torch.randn(2, 4, 64, 64, device="cpu")
        out = net(x)
        assert out.device.type == "cpu"
        assert out.shape == (2, 4)

    def test_forward_backward_gradient_flow(self):
        """9. Verify forward pass followed by backward() produces valid gradients on all parameters."""
        net = DuelingDQN()
        torch.manual_seed(42)
        x = torch.randn(2, 4, 64, 64, requires_grad=False)
        targets = torch.randn(2, 4, requires_grad=False)

        q = net(x)
        # Use MSE loss against action targets so non-uniform action gradients flow through Advantage stream
        loss = torch.nn.functional.mse_loss(q, targets)
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.isfinite(param.grad).all().item(), f"Parameter {name} has non-finite gradient"
            assert not torch.all(param.grad == 0), f"Parameter {name} gradient is completely zero"
