"""
Federated Learning Round Runner.

Implements synchronous FL rounds with:
- Participation sampling (100%, 70%, 50%)
- Round timing
- Metric collection and logging
"""
import time
from typing import Dict, List, Optional, Any
from collections import OrderedDict

import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import FLServer
from client import FLClient, create_clients_from_datasets
from data.client_dataset import ClientDataset, compute_all_clients_non_iid_stats
from utils.logging import ExperimentLogger


class FLRunner:
    """
    Federated Learning Round Runner.
    
    Coordinates server and clients for synchronous FL training.
    """
    
    def __init__(
        self,
        server: FLServer,
        clients: Dict[int, FLClient],
        logger: Optional[ExperimentLogger] = None,
        device: str = 'cpu'
    ):
        """
        Initialize FL runner.
        
        Args:
            server: FLServer instance
            clients: Dict mapping client_id to FLClient
            logger: ExperimentLogger for metrics (optional)
            device: Device to use
        """
        self.server = server
        self.clients = clients
        self.logger = logger
        self.device = device
        
        # Register client IDs with server
        self.server.set_client_ids(list(clients.keys()))
        
        # Track round metrics
        self.round_times: List[float] = []
        self.round_metrics: List[Dict[str, Any]] = []
    
    def run_round(
        self,
        round_num: int,
        participation_fraction: float = 1.0,
        algorithm: str = 'fedavg'
    ) -> Dict[str, Any]:
        """
        Run a single FL round.
        
        Args:
            round_num: Current round number
            participation_fraction: Fraction of clients to participate
            algorithm: Aggregation algorithm ('fedavg', 'fedprox', 'scaffold')
        
        Returns:
            Dict with round metrics
        """
        round_start = time.time()
        
        # 1. Select participating clients
        selected_ids = self.server.select_clients(participation_fraction)
        
        # 2. Get global model weights
        global_weights = self.server.get_global_weights()
        
        # Get server control for SCAFFOLD
        server_control = None
        if algorithm == 'scaffold' and hasattr(self.server, 'server_control'):
            server_control = self.server.server_control
        
        # 3. Client local training
        client_updates = []
        client_metrics = []
        
        for client_id in selected_ids:
            client = self.clients[client_id]
            
            # Send global model to client (with server control for SCAFFOLD)
            if algorithm == 'scaffold' and hasattr(client, 'receive_global_model'):
                # SCAFFOLD clients need server control
                client.receive_global_model(
                    global_weights, 
                    self.server.input_channels,
                    server_control=server_control
                )
            else:
                client.receive_global_model(global_weights, self.server.input_channels)
            
            # Local training
            train_metrics = client.train_local()
            
            # Evaluate on validation set
            val_metrics = client.evaluate()
            
            # Collect update (SCAFFOLD clients return extra control variate)
            client_updates.append(client.get_update())
            
            # Collect metrics for logging
            client_metrics.append({
                'client_id': client_id,
                'participation_flag': True,
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_metrics['val_loss'],
                'rmse': val_metrics['rmse'],
                'mae': val_metrics['mae'],
                'num_samples': client.num_samples,
            })
        
        # Log non-participating clients too
        # IMPORTANT: Preserve client model state to avoid leakage for algorithms
        # like FedProx/SCAFFOLD that maintain local state
        non_participating = set(self.clients.keys()) - set(selected_ids)
        for client_id in non_participating:
            client = self.clients[client_id]
            
            # Backup current client state (if any) to prevent leakage
            state_backup = None
            if client.model is not None:
                state_backup = client.get_model_weights()
            
            # Also backup algorithm-specific state
            control_backup = None
            if hasattr(client, 'client_control') and client.client_control is not None:
                import copy
                control_backup = copy.deepcopy(client.client_control)
            
            # Evaluate with current global model
            client.receive_global_model(global_weights, self.server.input_channels)
            val_metrics = client.evaluate()
            
            # Restore client state to prevent algorithm state leakage
            if state_backup is not None:
                client.model.load_state_dict(state_backup)
            if control_backup is not None:
                client.client_control = control_backup
            
            client_metrics.append({
                'client_id': client_id,
                'participation_flag': False,
                'train_loss': 0.0,  # Did not train
                'val_loss': val_metrics['val_loss'],
                'rmse': val_metrics['rmse'],
                'mae': val_metrics['mae'],
                'num_samples': client.num_samples,
            })
        
        # 4. Aggregate updates on server
        if client_updates:
            if algorithm == 'scaffold':
                # SCAFFOLD has special aggregation
                aggregated, _ = self.server.aggregate_scaffold(client_updates)
            elif algorithm == 'feddc':
                # FedDC has special aggregation with drift variables
                aggregated, _ = self.server.aggregate_feddc(client_updates)
            else:
                # FedAvg, FedProx use standard aggregation
                aggregated = self.server.aggregate_updates(client_updates, algorithm)
            
            # 5. Apply aggregated update
            self.server.apply_update(aggregated)
        
        round_time = time.time() - round_start
        self.round_times.append(round_time)
        
        # Calculate global metrics (weighted by sample count for correct aggregation)
        participating_metrics = [m for m in client_metrics if m['participation_flag']]
        
        # Weighted RMSE and MAE: weight by number of samples per client
        # This gives the correct global metric over all samples
        total_samples = sum(m['num_samples'] for m in client_metrics if m['rmse'] > 0)
        if total_samples > 0:
            # Weighted sum of squared errors for RMSE
            weighted_mse = sum(m['rmse']**2 * m['num_samples'] for m in client_metrics if m['rmse'] > 0)
            global_rmse = (weighted_mse / total_samples) ** 0.5
            # Weighted MAE
            weighted_mae = sum(m['mae'] * m['num_samples'] for m in client_metrics if m['mae'] > 0)
            global_mae = weighted_mae / total_samples
        else:
            global_rmse = 0.0
            global_mae = 0.0
        
        # Weighted average train loss by participating client samples
        participating_samples = sum(m['num_samples'] for m in participating_metrics)
        avg_train_loss = sum(m['train_loss'] * m['num_samples'] for m in participating_metrics) / participating_samples if participating_samples > 0 else 0.0
        
        round_summary = {
            'round': round_num,
            'round_time': round_time,
            'n_participating': len(selected_ids),
            'n_total': len(self.clients),
            'participation_fraction': participation_fraction,
            'global_rmse': global_rmse,
            'global_mae': global_mae,
            'avg_train_loss': avg_train_loss,
            'client_metrics': client_metrics,
        }
        
        self.round_metrics.append(round_summary)
        
        # Log to ExperimentLogger if available
        if self.logger:
            self.logger.log_round_batch(round_num, client_metrics)
        
        return round_summary
    
    def run_training(
        self,
        max_rounds: int = 100,
        participation_fraction: float = 1.0,
        algorithm: str = 'fedavg',
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run full FL training.
        
        Args:
            max_rounds: Maximum number of rounds
            participation_fraction: Client participation fraction
            algorithm: Aggregation algorithm
            verbose: Print progress
        
        Returns:
            List of round metrics
        """
        if verbose:
            print(f"Starting FL training: {max_rounds} rounds, "
                  f"{participation_fraction*100:.0f}% participation, {algorithm}")
        
        for round_num in range(max_rounds):
            round_metrics = self.run_round(round_num, participation_fraction, algorithm)
            
            if verbose and (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}/{max_rounds}: "
                      f"RMSE={round_metrics['global_rmse']:.2f}, "
                      f"MAE={round_metrics['global_mae']:.2f}, "
                      f"Time={round_metrics['round_time']:.2f}s")
        
        if verbose:
            total_time = sum(self.round_times)
            print(f"Training complete. Total time: {total_time:.1f}s")
        
        return self.round_metrics
    
    def evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate global model on all clients' test data.
        
        Returns:
            Dict with test_rmse, test_mae
        """
        global_weights = self.server.get_global_weights()
        
        all_rmse = []
        all_mae = []
        total_samples = 0
        
        for client_id, client in self.clients.items():
            client.receive_global_model(global_weights, self.server.input_channels)
            test_loader = client.dataset.get_test_loader()
            
            if test_loader:
                metrics = client.evaluate(test_loader)
                n_samples = client.dataset.test_size
                all_rmse.append(metrics['rmse'] * n_samples)
                all_mae.append(metrics['mae'] * n_samples)
                total_samples += n_samples
        
        test_rmse = sum(all_rmse) / total_samples if total_samples > 0 else 0.0
        test_mae = sum(all_mae) / total_samples if total_samples > 0 else 0.0
        
        return {
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'n_test_samples': total_samples
        }


def create_fl_runner(
    client_datasets: Dict[int, ClientDataset],
    input_channels: int = 17,
    device: str = 'cpu',
    seed: Optional[int] = None,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    local_epochs: int = 1,
    logger: Optional[ExperimentLogger] = None
) -> FLRunner:
    """
    Create a complete FL runner with server and clients.
    
    Args:
        client_datasets: Dict mapping client_id to ClientDataset
        input_channels: Number of input features
        device: Device to use
        seed: Random seed
        learning_rate: Learning rate (1e-3 per spec)
        batch_size: Batch size (64 per spec)
        local_epochs: Local epochs per round (1 per spec)
        logger: ExperimentLogger instance
    
    Returns:
        Configured FLRunner
    """
    # Create server
    server = FLServer(
        input_channels=input_channels,
        device=device,
        seed=seed
    )
    server.init_model()
    
    # Create clients
    clients = create_clients_from_datasets(
        client_datasets,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        local_epochs=local_epochs
    )
    
    # Create runner
    runner = FLRunner(
        server=server,
        clients=clients,
        logger=logger,
        device=device
    )
    
    return runner
