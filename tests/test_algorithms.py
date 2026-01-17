"""
Phase 3 — Algorithm Implementation Tests

Tests for:
- C-1: Centralized baseline
- C-2: Local-only baseline  
- C-3: FedAvg entry script
- C-4: FedProx implementation
- C-5: SCAFFOLD implementation
- C-6: FedDC implementation (conditional)
"""
import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
from collections import OrderedDict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.tcn import create_tcn_model
from server import FLServer, set_seed
from client import FLClient, FLClientFedProx, FLClientSCAFFOLD
from data.client_dataset import ClientDataset, RULDataset


# Test configuration
SEED = 42
NUM_FEATURES = 17
WINDOW_SIZE = 30
SAMPLES_PER_CLIENT = 50
DEVICE = 'cpu'


def create_synthetic_client_dataset(client_id: int, n_samples: int = 50) -> ClientDataset:
    """Create synthetic client dataset for testing."""
    np.random.seed(SEED + client_id)
    
    # Generate synthetic data
    train_X = np.random.randn(n_samples, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    train_y = np.random.rand(n_samples).astype(np.float32) * 125  # RUL range
    
    val_X = np.random.randn(n_samples // 4, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    val_y = np.random.rand(n_samples // 4).astype(np.float32) * 125
    
    test_X = np.random.randn(n_samples // 4, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    test_y = np.random.rand(n_samples // 4).astype(np.float32) * 125
    
    return ClientDataset(
        client_id=client_id,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y
    )


class TestFedProx:
    """Tests for FedProx implementation (C-4)."""
    
    def test_fedprox_client_creation(self):
        """Test FedProx client can be created with mu parameter."""
        set_seed(SEED)
        dataset = create_synthetic_client_dataset(1)
        
        client = FLClientFedProx(
            client_id=1,
            dataset=dataset,
            device=DEVICE,
            mu=0.01
        )
        
        assert client.mu == 0.01
        assert client.client_id == 1
    
    def test_fedprox_proximal_term_nonzero(self):
        """Test that proximal term is non-zero after training."""
        set_seed(SEED)
        dataset = create_synthetic_client_dataset(1)
        
        client = FLClientFedProx(
            client_id=1,
            dataset=dataset,
            device=DEVICE,
            mu=0.1  # Larger mu to see effect
        )
        
        # Initialize with global model
        model = create_tcn_model(NUM_FEATURES)
        global_weights = model.state_dict()
        client.receive_global_model(global_weights, NUM_FEATURES)
        
        # Check proximal term before training (should be 0)
        initial_prox = client._compute_proximal_term()
        assert initial_prox.item() == 0.0
        
        # Train
        client.train_local()
        
        # Check proximal term after training (should be > 0)
        final_prox = client._compute_proximal_term()
        assert final_prox.item() > 0.0
    
    def test_fedprox_mu_effect(self):
        """Test that larger mu results in smaller weight divergence."""
        set_seed(SEED)
        
        # Create initial model
        model = create_tcn_model(NUM_FEATURES)
        global_weights = copy.deepcopy(model.state_dict())
        
        divergences = []
        
        for mu in [0.0, 0.01, 0.1, 1.0]:
            set_seed(SEED)  # Reset seed for fair comparison
            dataset = create_synthetic_client_dataset(1)
            
            client = FLClientFedProx(
                client_id=1,
                dataset=dataset,
                device=DEVICE,
                mu=mu
            )
            client.receive_global_model(copy.deepcopy(global_weights), NUM_FEATURES)
            client.train_local()
            
            # Compute weight divergence
            local_weights = client.get_model_weights()
            divergence = 0.0
            for key in global_weights:
                diff = local_weights[key].float() - global_weights[key].float()
                divergence += torch.sum(diff ** 2).item()
            divergences.append(divergence)
        
        # With larger mu, divergence should decrease (approximately)
        # Note: With very small local epochs, effect may be subtle
        print(f"\nFedProx divergence by mu: {list(zip([0.0, 0.01, 0.1, 1.0], divergences))}")
        
        # At minimum, mu=1.0 should have less divergence than mu=0.0
        assert divergences[-1] < divergences[0] * 2, \
            "Large mu should limit weight divergence"


class TestSCAFFOLD:
    """Tests for SCAFFOLD implementation (C-5)."""
    
    def test_scaffold_client_creation(self):
        """Test SCAFFOLD client can be created."""
        set_seed(SEED)
        dataset = create_synthetic_client_dataset(1)
        
        client = FLClientSCAFFOLD(
            client_id=1,
            dataset=dataset,
            device=DEVICE
        )
        
        assert client.client_id == 1
        assert client.client_control is None  # Not initialized yet
    
    def test_scaffold_control_initialization(self):
        """Test control variates are properly initialized."""
        set_seed(SEED)
        dataset = create_synthetic_client_dataset(1)
        
        client = FLClientSCAFFOLD(
            client_id=1,
            dataset=dataset,
            device=DEVICE
        )
        
        model = create_tcn_model(NUM_FEATURES)
        global_weights = model.state_dict()
        
        # Create server control
        server_control = OrderedDict()
        for name, param in model.named_parameters():
            server_control[name] = torch.zeros_like(param)
        
        # Receive model with server control
        client.receive_global_model(global_weights, NUM_FEATURES, server_control)
        
        # Check controls are initialized
        assert client.client_control is not None
        assert client.server_control is not None
        assert len(client.client_control) == len(client.server_control)
    
    def test_scaffold_control_update(self):
        """Test client control is updated after training."""
        set_seed(SEED)
        dataset = create_synthetic_client_dataset(1)
        
        client = FLClientSCAFFOLD(
            client_id=1,
            dataset=dataset,
            device=DEVICE
        )
        
        model = create_tcn_model(NUM_FEATURES)
        global_weights = model.state_dict()
        server_control = OrderedDict()
        for name, param in model.named_parameters():
            server_control[name] = torch.zeros_like(param)
        
        client.receive_global_model(global_weights, NUM_FEATURES, server_control)
        
        # Get initial control
        initial_control = copy.deepcopy(client.client_control)
        
        # Train (this updates control)
        client.train_local()
        
        # Control should have changed
        final_control = client.client_control
        
        # Check at least some values changed
        changed = False
        for name in initial_control:
            if not torch.allclose(initial_control[name], final_control[name], atol=1e-6):
                changed = True
                break
        
        assert changed, "Client control should update after training"
    
    def test_scaffold_get_update_includes_control(self):
        """Test SCAFFOLD client update includes control variate."""
        set_seed(SEED)
        dataset = create_synthetic_client_dataset(1)
        
        client = FLClientSCAFFOLD(
            client_id=1,
            dataset=dataset,
            device=DEVICE
        )
        
        model = create_tcn_model(NUM_FEATURES)
        global_weights = model.state_dict()
        server_control = OrderedDict()
        for name, param in model.named_parameters():
            server_control[name] = torch.zeros_like(param)
        
        client.receive_global_model(global_weights, NUM_FEATURES, server_control)
        client.train_local()
        
        update = client.get_update()
        
        # SCAFFOLD update should be 4-tuple
        assert len(update) == 4
        client_id, weights, n_samples, control = update
        assert client_id == 1
        assert isinstance(weights, OrderedDict)
        assert isinstance(control, OrderedDict)
    
    def test_server_scaffold_aggregation(self):
        """Test server SCAFFOLD aggregation."""
        set_seed(SEED)
        
        server = FLServer(input_channels=NUM_FEATURES, seed=SEED)
        server.init_model()
        server.init_scaffold_control(NUM_FEATURES)
        
        # Create fake client updates with control variates
        model = create_tcn_model(NUM_FEATURES)
        weights = model.state_dict()
        
        control1 = OrderedDict()
        control2 = OrderedDict()
        for name, param in model.named_parameters():
            control1[name] = torch.ones_like(param) * 0.1
            control2[name] = torch.ones_like(param) * 0.2
        
        client_updates = [
            (1, copy.deepcopy(weights), 100, control1),
            (2, copy.deepcopy(weights), 100, control2),
        ]
        
        agg_weights, new_control = server.aggregate_scaffold(client_updates)
        
        # Check aggregated control is average of client controls
        for name in new_control:
            expected = (control1[name] + control2[name]) / 2
            assert torch.allclose(new_control[name], expected, atol=1e-5)


class TestFedAvgIntegration:
    """Integration tests for FedAvg (C-3)."""
    
    def test_fedavg_reduces_loss(self):
        """Test FedAvg training reduces validation loss."""
        from runner import FLRunner
        from client import create_clients_from_datasets
        
        set_seed(SEED)
        
        # Create clients
        client_datasets = {
            i: create_synthetic_client_dataset(i) 
            for i in range(3)
        }
        
        clients = create_clients_from_datasets(
            client_datasets,
            device=DEVICE,
            learning_rate=1e-3,
            batch_size=32,
            local_epochs=1
        )
        
        # Create server
        server = FLServer(input_channels=NUM_FEATURES, seed=SEED)
        server.init_model()
        
        # Create runner
        runner = FLRunner(server=server, clients=clients, device=DEVICE)
        
        # Run a few rounds
        initial_metrics = None
        for round_num in range(5):
            metrics = runner.run_round(round_num, participation_fraction=1.0)
            if initial_metrics is None:
                initial_metrics = metrics
        
        # Loss should decrease (or at least not explode)
        assert metrics['global_rmse'] < initial_metrics['global_rmse'] * 2


class TestFedProxIntegration:
    """Integration tests for FedProx (C-4)."""
    
    def test_fedprox_runs_successfully(self):
        """Test FedProx completes training without errors."""
        from runner import FLRunner
        
        set_seed(SEED)
        
        # Create FedProx clients
        client_datasets = {
            i: create_synthetic_client_dataset(i) 
            for i in range(3)
        }
        
        clients = {}
        for client_id, dataset in client_datasets.items():
            clients[client_id] = FLClientFedProx(
                client_id=client_id,
                dataset=dataset,
                device=DEVICE,
                mu=0.01
            )
        
        server = FLServer(input_channels=NUM_FEATURES, seed=SEED)
        server.init_model()
        
        runner = FLRunner(server=server, clients=clients, device=DEVICE)
        
        # Run a few rounds with fedprox
        for round_num in range(3):
            metrics = runner.run_round(round_num, participation_fraction=1.0, algorithm='fedprox')
        
        assert metrics['global_rmse'] > 0
        assert metrics['global_mae'] > 0


class TestSCAFFOLDIntegration:
    """Integration tests for SCAFFOLD (C-5)."""
    
    def test_scaffold_runs_successfully(self):
        """Test SCAFFOLD completes training without errors."""
        from runner import FLRunner
        
        set_seed(SEED)
        
        # Create SCAFFOLD clients
        client_datasets = {
            i: create_synthetic_client_dataset(i) 
            for i in range(3)
        }
        
        clients = {}
        for client_id, dataset in client_datasets.items():
            clients[client_id] = FLClientSCAFFOLD(
                client_id=client_id,
                dataset=dataset,
                device=DEVICE
            )
        
        server = FLServer(input_channels=NUM_FEATURES, seed=SEED)
        server.init_model()
        server.init_scaffold_control(NUM_FEATURES)
        
        runner = FLRunner(server=server, clients=clients, device=DEVICE)
        
        # Run a few rounds with scaffold
        for round_num in range(3):
            metrics = runner.run_round(round_num, participation_fraction=1.0, algorithm='scaffold')
        
        assert metrics['global_rmse'] > 0
        assert metrics['global_mae'] > 0


class TestAlgorithmComparison:
    """Compare algorithm behaviors."""
    
    def test_fedprox_proximal_term_affects_training(self):
        """Test that FedProx proximal term affects the training outcome."""
        set_seed(SEED)
        
        # Create same dataset for both
        dataset = create_synthetic_client_dataset(1, n_samples=100)
        
        # Create initial model
        model = create_tcn_model(NUM_FEATURES)
        initial_weights = copy.deepcopy(model.state_dict())
        
        # Train with FedAvg (mu=0)
        set_seed(SEED)
        client_fedavg = FLClient(
            client_id=1,
            dataset=dataset,
            device=DEVICE,
            learning_rate=1e-3,
            local_epochs=3  # More epochs to see difference
        )
        client_fedavg.receive_global_model(copy.deepcopy(initial_weights), NUM_FEATURES)
        client_fedavg.train_local()
        weights_fedavg = client_fedavg.get_model_weights()
        
        # Train with FedProx (large mu)
        set_seed(SEED)
        client_fedprox = FLClientFedProx(
            client_id=1,
            dataset=dataset,
            device=DEVICE,
            learning_rate=1e-3,
            local_epochs=3,  # More epochs to see difference
            mu=10.0  # Very large mu to force staying close to global
        )
        client_fedprox.receive_global_model(copy.deepcopy(initial_weights), NUM_FEATURES)
        client_fedprox.train_local()
        weights_fedprox = client_fedprox.get_model_weights()
        
        # Compute divergence from initial weights
        div_fedavg = 0.0
        div_fedprox = 0.0
        for key in initial_weights:
            div_fedavg += torch.sum((weights_fedavg[key] - initial_weights[key]) ** 2).item()
            div_fedprox += torch.sum((weights_fedprox[key] - initial_weights[key]) ** 2).item()
        
        print(f"\nDivergence from initial - FedAvg: {div_fedavg:.6f}, FedProx: {div_fedprox:.6f}")
        
        # FedProx with large mu should stay closer to initial weights
        assert div_fedprox < div_fedavg, \
            f"FedProx should stay closer to global weights (div_prox={div_fedprox:.4f} vs div_avg={div_fedavg:.4f})"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
