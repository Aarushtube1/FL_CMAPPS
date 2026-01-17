"""
Federated Learning Server.

Implements the FL server with:
- init_model(): Initialize global model
- select_clients(): Sample clients for participation
- aggregate_updates(): Aggregate client updates (FedAvg by default)
- apply_update(): Apply aggregated update to global model
"""
import copy
import random
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.tcn import create_tcn_model, count_parameters


def set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU, safe to call even on CPU
    # For full determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FLServer:
    """
    Federated Learning Server.
    
    Manages global model, client selection, and update aggregation.
    """
    
    def __init__(
        self,
        input_channels: int = 17,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """
        Initialize FL server.
        
        Args:
            input_channels: Number of input features for TCN
            device: Device to use ('cpu' or 'cuda')
            seed: Random seed for reproducibility
        """
        self.device = device
        self.seed = seed
        
        if seed is not None:
            set_seed(seed)
        
        # Initialize global model
        self.global_model = None
        self.input_channels = input_channels
        self.current_round = 0
        
        # Track participating clients
        self.all_client_ids: List[int] = []
    
    def init_model(self) -> nn.Module:
        """
        Initialize the global model.
        
        Returns:
            Initialized TCN model
        """
        self.global_model = create_tcn_model(self.input_channels)
        self.global_model.to(self.device)
        
        print(f"Initialized global TCN model with {count_parameters(self.global_model):,} parameters")
        return self.global_model
    
    def get_global_weights(self) -> OrderedDict:
        """Get current global model weights."""
        if self.global_model is None:
            raise RuntimeError("Global model not initialized. Call init_model() first.")
        return copy.deepcopy(self.global_model.state_dict())
    
    def set_client_ids(self, client_ids: List[int]):
        """Register available client IDs."""
        self.all_client_ids = list(client_ids)
    
    def select_clients(
        self,
        participation_fraction: float = 1.0,
        min_clients: int = 1
    ) -> List[int]:
        """
        Select clients for the current round.
        
        Args:
            participation_fraction: Fraction of clients to select (0.0-1.0)
            min_clients: Minimum number of clients to select
        
        Returns:
            List of selected client IDs
        """
        if not self.all_client_ids:
            raise RuntimeError("No client IDs registered. Call set_client_ids() first.")
        
        n_total = len(self.all_client_ids)
        n_select = max(min_clients, int(n_total * participation_fraction))
        n_select = min(n_select, n_total)  # Can't select more than available
        
        selected = random.sample(self.all_client_ids, n_select)
        return sorted(selected)
    
    def aggregate_updates(
        self,
        client_updates: List[Tuple[int, OrderedDict, int]],
        algorithm: str = 'fedavg'
    ) -> OrderedDict:
        """
        Aggregate client updates.
        
        Args:
            client_updates: List of (client_id, state_dict, num_samples) tuples
            algorithm: Aggregation algorithm ('fedavg', 'fedprox', 'scaffold', 'feddc')
        
        Returns:
            Aggregated state dict
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        if algorithm in ('fedavg', 'fedprox'):
            # FedProx uses same aggregation as FedAvg
            # (proximal term only affects local training, not aggregation)
            return self._fedavg_aggregate(client_updates)
        elif algorithm == 'scaffold':
            # SCAFFOLD aggregation handled separately (needs control variates)
            raise ValueError("Use aggregate_scaffold() for SCAFFOLD algorithm")
        elif algorithm == 'feddc':
            # FedDC aggregation handled separately (needs drift variables)
            raise ValueError("Use aggregate_feddc() for FedDC algorithm")
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not yet implemented")
    
    def _fedavg_aggregate(
        self,
        client_updates: List[Tuple[int, OrderedDict, int]]
    ) -> OrderedDict:
        """
        FedAvg aggregation: weighted average by number of samples.
        
        w_global = sum(n_k * w_k) / sum(n_k)
        """
        # Calculate total samples
        total_samples = sum(n_samples for _, _, n_samples in client_updates)
        
        # Initialize aggregated weights
        aggregated = OrderedDict()
        
        # Get first client's state dict as template
        _, first_state, _ = client_updates[0]
        for key in first_state.keys():
            aggregated[key] = torch.zeros_like(first_state[key], dtype=torch.float32)
        
        # Weighted sum
        for client_id, state_dict, n_samples in client_updates:
            weight = n_samples / total_samples
            for key in state_dict.keys():
                aggregated[key] += weight * state_dict[key].float()
        
        return aggregated
    
    def init_scaffold_control(self, input_channels: int = 17):
        """Initialize server control variate for SCAFFOLD (C-5)."""
        if self.global_model is None:
            self.init_model()
        
        self.server_control = OrderedDict()
        for name, param in self.global_model.named_parameters():
            self.server_control[name] = torch.zeros_like(param, device='cpu')
    
    def aggregate_scaffold(
        self,
        client_updates: List[Tuple[int, OrderedDict, int, OrderedDict]]
    ) -> Tuple[OrderedDict, OrderedDict]:
        """
        SCAFFOLD aggregation: aggregate weights and control variates.
        
        Args:
            client_updates: List of (client_id, weights, n_samples, control_delta) tuples
        
        Returns:
            (aggregated_weights, new_server_control)
        """
        n_clients = len(client_updates)
        
        # Aggregate weights (same as FedAvg)
        weight_updates = [(cid, w, n) for cid, w, n, _ in client_updates]
        aggregated_weights = self._fedavg_aggregate(weight_updates)
        
        # Aggregate control variates: c = c + (1/N) * sum(delta_c_i)
        # where delta_c_i = c_i_new - c_i_old (but clients send c_i_new)
        if hasattr(self, 'server_control') and self.server_control is not None:
            for name in self.server_control.keys():
                # Average client controls
                avg_control = torch.zeros_like(self.server_control[name])
                for _, _, _, control in client_updates:
                    if name in control:
                        avg_control += control[name].float()
                avg_control /= n_clients
                self.server_control[name] = avg_control
        
        return aggregated_weights, self.server_control
    
    def init_feddc_drift(self, input_channels: int = 17):
        """Initialize server-side drift tracking for FedDC (C-6)."""
        if self.global_model is None:
            self.init_model()
        
        self.global_drift = OrderedDict()
        for name, param in self.global_model.named_parameters():
            self.global_drift[name] = torch.zeros_like(param, device='cpu')
    
    def aggregate_feddc(
        self,
        client_updates: List[Tuple[int, OrderedDict, int, OrderedDict]]
    ) -> Tuple[OrderedDict, OrderedDict]:
        """
        FedDC aggregation: aggregate weights and drift variables.
        
        Args:
            client_updates: List of (client_id, weights, n_samples, drift) tuples
        
        Returns:
            (aggregated_weights, global_drift)
        """
        # Aggregate weights (same as FedAvg - weighted by samples)
        weight_updates = [(cid, w, n) for cid, w, n, _ in client_updates]
        aggregated_weights = self._fedavg_aggregate(weight_updates)
        
        # Aggregate drift variables (simple average for global drift estimation)
        n_clients = len(client_updates)
        if hasattr(self, 'global_drift') and self.global_drift is not None:
            for name in self.global_drift.keys():
                avg_drift = torch.zeros_like(self.global_drift[name])
                for _, _, _, drift in client_updates:
                    if name in drift:
                        avg_drift += drift[name].float()
                avg_drift /= n_clients
                self.global_drift[name] = avg_drift
        
        return aggregated_weights, self.global_drift

    def apply_update(self, aggregated_weights: OrderedDict):
        """
        Apply aggregated update to global model.
        
        Args:
            aggregated_weights: Aggregated state dict from aggregate_updates()
        """
        if self.global_model is None:
            raise RuntimeError("Global model not initialized. Call init_model() first.")
        
        self.global_model.load_state_dict(aggregated_weights)
        self.current_round += 1
    
    def get_model_for_client(self) -> nn.Module:
        """
        Get a copy of the global model for a client.
        
        Returns:
            Copy of global model
        """
        if self.global_model is None:
            raise RuntimeError("Global model not initialized. Call init_model() first.")
        
        client_model = create_tcn_model(self.input_channels)
        client_model.load_state_dict(self.get_global_weights())
        return client_model
    
    def save_checkpoint(self, path: str):
        """Save server checkpoint."""
        checkpoint = {
            'current_round': self.current_round,
            'global_model_state': self.global_model.state_dict() if self.global_model else None,
            'all_client_ids': self.all_client_ids,
            'seed': self.seed,
            'input_channels': self.input_channels,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load server checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_round = checkpoint['current_round']
        self.all_client_ids = checkpoint['all_client_ids']
        self.seed = checkpoint['seed']
        self.input_channels = checkpoint['input_channels']
        
        if checkpoint['global_model_state'] is not None:
            if self.global_model is None:
                self.init_model()
            self.global_model.load_state_dict(checkpoint['global_model_state'])
