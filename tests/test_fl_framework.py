"""
Smoke tests for FL Framework (Phase 2).

Tests:
- Server API: init_model, select_clients, aggregate_updates, apply_update
- Client: local training, weight serialization
- Runner: round execution with participation fractions
"""
import os
import sys
import numpy as np
import torch
import pytest

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from models.tcn import TCN, create_tcn_model, count_parameters
from server import FLServer
from client import FLClient, create_clients_from_datasets
from runner import FLRunner, create_fl_runner
from data.client_dataset import ClientDataset


# Fixtures for mock data
@pytest.fixture
def mock_client_datasets():
    """Create mock ClientDataset objects for testing."""
    datasets = {}
    for client_id in range(1, 6):  # 5 mock clients
        n_train = 50 + client_id * 10
        n_val = 10
        n_test = 10
        n_features = 17
        seq_len = 30
        
        train_X = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
        train_y = np.random.rand(n_train).astype(np.float32) * 125  # RUL 0-125
        val_X = np.random.randn(n_val, seq_len, n_features).astype(np.float32)
        val_y = np.random.rand(n_val).astype(np.float32) * 125
        test_X = np.random.randn(n_test, seq_len, n_features).astype(np.float32)
        test_y = np.random.rand(n_test).astype(np.float32) * 125
        
        datasets[client_id] = ClientDataset(
            client_id=client_id,
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y
        )
    return datasets


class TestTCNModel:
    """Tests for TCN model."""
    
    def test_model_creation(self):
        """Model should be created with correct architecture."""
        model = create_tcn_model(input_channels=17)
        assert isinstance(model, TCN)
        assert model.num_blocks == 3
        assert model.filters == 64
    
    def test_forward_pass(self):
        """Forward pass should produce correct output shape."""
        model = create_tcn_model(input_channels=17)
        batch = torch.randn(8, 30, 17)  # (batch, seq_len, features)
        output = model(batch)
        assert output.shape == (8,), f"Expected (8,), got {output.shape}"
    
    def test_parameter_count(self):
        """Model should have reasonable parameter count."""
        model = create_tcn_model(input_channels=17)
        n_params = count_parameters(model)
        assert n_params > 10000, "Model should have >10k parameters"
        assert n_params < 1000000, "Model should have <1M parameters"


class TestFLServer:
    """Tests for FL Server (A-1)."""
    
    def test_init_model(self):
        """Server should initialize global model."""
        server = FLServer(input_channels=17, seed=42)
        model = server.init_model()
        assert model is not None
        assert server.global_model is not None
    
    def test_select_clients(self):
        """Server should select correct fraction of clients."""
        server = FLServer(seed=42)
        server.init_model()
        server.set_client_ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 100% participation
        selected = server.select_clients(1.0)
        assert len(selected) == 10
        
        # 50% participation
        selected = server.select_clients(0.5)
        assert len(selected) == 5
        
        # 70% participation
        selected = server.select_clients(0.7)
        assert len(selected) == 7
    
    def test_aggregate_updates_fedavg(self):
        """FedAvg aggregation should compute weighted average."""
        server = FLServer(seed=42)
        server.init_model()
        
        # Create mock updates with known weights
        weights1 = server.get_global_weights()
        weights2 = server.get_global_weights()
        
        # Modify weights slightly
        for key in weights1:
            weights1[key] = weights1[key] * 0.8
            weights2[key] = weights2[key] * 1.2
        
        updates = [
            (1, weights1, 100),  # 100 samples
            (2, weights2, 100),  # 100 samples
        ]
        
        aggregated = server.aggregate_updates(updates, 'fedavg')
        
        # With equal samples, should be average
        original = server.get_global_weights()
        for key in aggregated:
            expected = (weights1[key] + weights2[key]) / 2
            assert torch.allclose(aggregated[key], expected, atol=1e-5)
    
    def test_apply_update(self):
        """Server should apply aggregated weights."""
        server = FLServer(seed=42)
        server.init_model()
        
        original = server.get_global_weights()
        
        # Modify and apply
        modified = {k: v * 2 for k, v in original.items()}
        server.apply_update(modified)
        
        new_weights = server.get_global_weights()
        for key in new_weights:
            assert torch.allclose(new_weights[key], original[key] * 2)


class TestFLClient:
    """Tests for FL Client (A-2)."""
    
    def test_client_creation(self, mock_client_datasets):
        """Client should be created correctly."""
        dataset = mock_client_datasets[1]
        client = FLClient(
            client_id=1,
            dataset=dataset,
            learning_rate=1e-3,
            batch_size=64,
            local_epochs=1
        )
        assert client.client_id == 1
        assert client.num_samples == dataset.train_size
    
    def test_receive_global_model(self, mock_client_datasets):
        """Client should receive and load global weights."""
        dataset = mock_client_datasets[1]
        client = FLClient(client_id=1, dataset=dataset)
        
        server = FLServer(seed=42)
        server.init_model()
        
        client.receive_global_model(server.get_global_weights())
        assert client.model is not None
        assert client.optimizer is not None
    
    def test_train_local(self, mock_client_datasets):
        """Client should train locally and return metrics."""
        dataset = mock_client_datasets[1]
        client = FLClient(client_id=1, dataset=dataset, local_epochs=1)
        
        server = FLServer(seed=42)
        server.init_model()
        client.receive_global_model(server.get_global_weights())
        
        metrics = client.train_local()
        assert 'train_loss' in metrics
        assert metrics['train_loss'] >= 0
    
    def test_evaluate(self, mock_client_datasets):
        """Client should evaluate and return metrics."""
        dataset = mock_client_datasets[1]
        client = FLClient(client_id=1, dataset=dataset)
        
        server = FLServer(seed=42)
        server.init_model()
        client.receive_global_model(server.get_global_weights())
        
        metrics = client.evaluate()
        assert 'val_loss' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
    
    def test_weight_serialization(self, mock_client_datasets):
        """Client should serialize weights correctly."""
        dataset = mock_client_datasets[1]
        client = FLClient(client_id=1, dataset=dataset)
        
        server = FLServer(seed=42)
        server.init_model()
        client.receive_global_model(server.get_global_weights())
        
        weights = client.get_model_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
    
    def test_get_update_tuple(self, mock_client_datasets):
        """get_update should return correct tuple format."""
        dataset = mock_client_datasets[1]
        client = FLClient(client_id=1, dataset=dataset)
        
        server = FLServer(seed=42)
        server.init_model()
        client.receive_global_model(server.get_global_weights())
        
        update = client.get_update()
        assert len(update) == 3
        assert update[0] == 1  # client_id
        assert isinstance(update[1], dict)  # state_dict
        assert update[2] == client.num_samples


class TestFLRunner:
    """Tests for FL Runner (A-3)."""
    
    def test_runner_creation(self, mock_client_datasets):
        """Runner should be created correctly."""
        runner = create_fl_runner(
            mock_client_datasets,
            seed=42
        )
        assert runner.server is not None
        assert len(runner.clients) == 5
    
    def test_run_single_round(self, mock_client_datasets):
        """Runner should execute a single round."""
        runner = create_fl_runner(mock_client_datasets, seed=42)
        
        metrics = runner.run_round(round_num=0, participation_fraction=1.0)
        
        assert 'round' in metrics
        assert metrics['round'] == 0
        assert 'global_rmse' in metrics
        assert 'global_mae' in metrics
        assert 'round_time' in metrics
        assert metrics['n_participating'] == 5
    
    def test_participation_fractions(self, mock_client_datasets):
        """Runner should respect participation fractions."""
        runner = create_fl_runner(mock_client_datasets, seed=42)
        
        # 100% participation
        metrics_100 = runner.run_round(0, participation_fraction=1.0)
        assert metrics_100['n_participating'] == 5
        
        # Create fresh runner for each test to reset state
        runner = create_fl_runner(mock_client_datasets, seed=42)
        metrics_60 = runner.run_round(0, participation_fraction=0.6)
        assert metrics_60['n_participating'] == 3  # 60% of 5 = 3
        
        runner = create_fl_runner(mock_client_datasets, seed=42)
        metrics_40 = runner.run_round(0, participation_fraction=0.4)
        assert metrics_40['n_participating'] == 2  # 40% of 5 = 2
    
    def test_round_timing(self, mock_client_datasets):
        """Runner should track round timing."""
        runner = create_fl_runner(mock_client_datasets, seed=42)
        
        metrics = runner.run_round(0, participation_fraction=1.0)
        
        assert metrics['round_time'] > 0
        assert len(runner.round_times) == 1
    
    def test_multiple_rounds(self, mock_client_datasets):
        """Runner should execute multiple rounds."""
        runner = create_fl_runner(mock_client_datasets, seed=42)
        
        all_metrics = runner.run_training(
            max_rounds=3,
            participation_fraction=1.0,
            verbose=False
        )
        
        assert len(all_metrics) == 3
        assert runner.server.current_round == 3
    
    def test_evaluate_global_model(self, mock_client_datasets):
        """Runner should evaluate global model on test data."""
        runner = create_fl_runner(mock_client_datasets, seed=42)
        runner.run_round(0, participation_fraction=1.0)
        
        test_metrics = runner.evaluate_global_model()
        
        assert 'test_rmse' in test_metrics
        assert 'test_mae' in test_metrics
        assert test_metrics['n_test_samples'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
