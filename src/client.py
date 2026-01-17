"""
Federated Learning Client.

Implements client-side training with:
- Local training loop (1 epoch per round, configurable)
- Adam optimizer (lr=1e-3)
- MSE loss
- Weight serialization for server communication

Supported algorithms:
- FLClient: Standard FedAvg client
- FLClientFedProx: FedProx with proximal regularization (C-4)
- FLClientSCAFFOLD: SCAFFOLD with control variates (C-5)
- FLClientFedDC: FedDC (C-6, conditional - placeholder)
"""
import copy
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.tcn import create_tcn_model
from data.client_dataset import ClientDataset, RULDataset


class FLClient:
    """
    Federated Learning Client.
    
    Manages local model training and communication with server.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset: ClientDataset,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        local_epochs: int = 1
    ):
        """
        Initialize FL client.
        
        Args:
            client_id: Unique client identifier
            dataset: ClientDataset with train/val/test data
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Learning rate for Adam optimizer (fixed at 1e-3 per spec)
            batch_size: Batch size for training (fixed at 64 per spec)
            local_epochs: Number of local epochs per round (fixed at 1 per spec)
        """
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        
        # Model will be set by receive_global_model
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        
        # Loss function (MSE per spec)
        self.criterion = nn.MSELoss()
        
        # Data loaders
        self.train_loader = dataset.get_train_loader(batch_size=batch_size, shuffle=True)
        self.val_loader = dataset.get_val_loader(batch_size=batch_size)
    
    @property
    def num_samples(self) -> int:
        """Number of training samples."""
        return self.dataset.train_size
    
    def receive_global_model(self, global_weights: OrderedDict, input_channels: int = 17):
        """
        Receive and load global model weights from server.
        
        Args:
            global_weights: State dict from server
            input_channels: Number of input features
        """
        if self.model is None:
            self.model = create_tcn_model(input_channels)
            self.model.to(self.device)
        
        self.model.load_state_dict(global_weights)
        
        # Reset optimizer with fresh state
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0  # Per spec
        )
    
    def train_local(self) -> Dict[str, float]:
        """
        Perform local training for specified epochs.
        
        Returns:
            Dict with training metrics (train_loss)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call receive_global_model() first.")
        
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_y)
                epoch_samples += len(batch_y)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return {'train_loss': avg_loss}
    
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on validation or test data.
        
        Args:
            loader: DataLoader to evaluate on (default: validation loader)
        
        Returns:
            Dict with val_loss, rmse, mae
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call receive_global_model() first.")
        
        if loader is None:
            loader = self.val_loader
        
        if loader is None:
            return {'val_loss': 0.0, 'rmse': 0.0, 'mae': 0.0}
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item() * len(batch_y)
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        n_samples = len(all_targets)
        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        
        # RMSE and MAE
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2)) if n_samples > 0 else 0.0
        mae = np.mean(np.abs(all_preds - all_targets)) if n_samples > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'rmse': float(rmse),
            'mae': float(mae)
        }
    
    def get_model_weights(self) -> OrderedDict:
        """
        Get current model weights for sending to server.
        
        Returns:
            State dict of local model
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call receive_global_model() first.")
        
        return copy.deepcopy(self.model.state_dict())
    
    def get_update(self) -> Tuple[int, OrderedDict, int]:
        """
        Get client update tuple for server aggregation.
        
        Returns:
            (client_id, state_dict, num_samples)
        """
        return (self.client_id, self.get_model_weights(), self.num_samples)
    
    def save_checkpoint(self, path: str):
        """Save client checkpoint."""
        checkpoint = {
            'client_id': self.client_id,
            'model_state': self.model.state_dict() if self.model else None,
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, input_channels: int = 17):
        """Load client checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if checkpoint['model_state'] is not None:
            if self.model is None:
                self.model = create_tcn_model(input_channels)
                self.model.to(self.device)
            self.model.load_state_dict(checkpoint['model_state'])
        
        if checkpoint['optimizer_state'] is not None:
            if self.optimizer is None:
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])


def create_clients_from_datasets(
    client_datasets: Dict[int, ClientDataset],
    device: str = 'cpu',
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    local_epochs: int = 1
) -> Dict[int, FLClient]:
    """
    Create FL clients from client datasets.
    
    Args:
        client_datasets: Dict mapping client_id to ClientDataset
        device: Device to use
        learning_rate: Learning rate (default 1e-3 per spec)
        batch_size: Batch size (default 64 per spec)
        local_epochs: Local epochs (default 1 per spec)
    
    Returns:
        Dict mapping client_id to FLClient
    """
    clients = {}
    for client_id, dataset in client_datasets.items():
        clients[client_id] = FLClient(
            client_id=client_id,
            dataset=dataset,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            local_epochs=local_epochs
        )
    return clients


class FLClientFedProx(FLClient):
    """
    FedProx Client (C-4).
    
    Adds proximal term to local objective:
    L_local = L_task + (mu/2) * ||w - w_global||^2
    
    This regularizes local updates to stay close to the global model,
    helping with non-IID data and partial participation.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset: ClientDataset,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        local_epochs: int = 1,
        mu: float = 0.01
    ):
        """
        Initialize FedProx client.
        
        Args:
            mu: Proximal term coefficient (default 0.01)
        """
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            local_epochs=local_epochs
        )
        self.mu = mu
        self.global_weights: Optional[OrderedDict] = None
    
    def receive_global_model(self, global_weights: OrderedDict, input_channels: int = 17):
        """Receive global model and store for proximal term."""
        super().receive_global_model(global_weights, input_channels)
        # Store global weights for proximal term calculation
        self.global_weights = copy.deepcopy(global_weights)
    
    def _compute_proximal_term(self) -> torch.Tensor:
        """Compute proximal term: (mu/2) * ||w - w_global||^2."""
        if self.global_weights is None:
            return torch.tensor(0.0, device=self.device)
        
        prox_term = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self.global_weights:
                global_param = self.global_weights[name].to(self.device)
                prox_term += torch.sum((param - global_param) ** 2)
        
        return (self.mu / 2.0) * prox_term
    
    def train_local(self) -> Dict[str, float]:
        """
        Perform local training with proximal term.
        
        L_total = L_task + (mu/2) * ||w - w_global||^2
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call receive_global_model() first.")
        
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                task_loss = self.criterion(predictions, batch_y)
                
                # Add proximal term
                prox_term = self._compute_proximal_term()
                loss = task_loss + prox_term
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += task_loss.item() * len(batch_y)
                epoch_samples += len(batch_y)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return {'train_loss': avg_loss}


class FLClientSCAFFOLD(FLClient):
    """
    SCAFFOLD Client (C-5).
    
    Uses control variates to correct for client drift:
    - c: server control variate (global)
    - c_i: client control variate (local)
    
    Local update with variance reduction:
    g_corrected = g - c_i + c
    
    This reduces the variance of client updates in non-IID settings.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset: ClientDataset,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        local_epochs: int = 1
    ):
        """Initialize SCAFFOLD client."""
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            local_epochs=local_epochs
        )
        # Control variates (initialized to zero)
        self.client_control: Optional[OrderedDict] = None
        self.server_control: Optional[OrderedDict] = None
        self.global_weights: Optional[OrderedDict] = None
    
    def receive_global_model(
        self, 
        global_weights: OrderedDict, 
        input_channels: int = 17,
        server_control: Optional[OrderedDict] = None
    ):
        """Receive global model and server control variate."""
        super().receive_global_model(global_weights, input_channels)
        self.global_weights = copy.deepcopy(global_weights)
        
        # Initialize or update server control
        if server_control is not None:
            self.server_control = copy.deepcopy(server_control)
        elif self.server_control is None:
            # Initialize to zeros
            self.server_control = OrderedDict()
            for name, param in self.model.named_parameters():
                self.server_control[name] = torch.zeros_like(param)
        
        # Initialize client control if needed
        if self.client_control is None:
            self.client_control = OrderedDict()
            for name, param in self.model.named_parameters():
                self.client_control[name] = torch.zeros_like(param)
    
    def train_local(self) -> Dict[str, float]:
        """
        Perform local training with SCAFFOLD correction.
        
        Gradient correction: g_corrected = g - c_i + c
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call receive_global_model() first.")
        
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                
                # Apply SCAFFOLD correction to gradients
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            # g_corrected = g - c_i + c
                            c_i = self.client_control[name].to(self.device)
                            c = self.server_control[name].to(self.device)
                            param.grad.data.add_(c - c_i)
                
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_y)
                epoch_samples += len(batch_y)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # Update client control variate
        self._update_client_control()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {'train_loss': avg_loss}
    
    def _update_client_control(self):
        """
        Update client control variate after local training.
        
        Option II from SCAFFOLD paper (more stable):
        c_i_new = c_i - c + (w_global - w_local) / (K * eta)
        
        where K = local_epochs * num_batches, eta = learning_rate
        """
        if self.global_weights is None:
            return
        
        # Estimate number of local steps
        K = self.local_epochs * len(self.train_loader)
        
        new_control = OrderedDict()
        for name, param in self.model.named_parameters():
            c_i = self.client_control[name].to(self.device)
            c = self.server_control[name].to(self.device)
            w_global = self.global_weights[name].to(self.device)
            w_local = param.data
            
            # c_i_new = c_i - c + (w_global - w_local) / (K * eta)
            delta = (w_global - w_local) / (K * self.learning_rate)
            new_control[name] = (c_i - c + delta).cpu()
        
        self.client_control = new_control
    
    def get_control_delta(self) -> OrderedDict:
        """
        Get client control variate delta for server aggregation.
        
        Returns delta_c_i = c_i_new - c_i_old (but we just return new c_i)
        """
        return copy.deepcopy(self.client_control)
    
    def get_update(self) -> Tuple[int, OrderedDict, int, OrderedDict]:
        """
        Get client update tuple including control variate.
        
        Returns:
            (client_id, state_dict, num_samples, control_delta)
        """
        return (
            self.client_id, 
            self.get_model_weights(), 
            self.num_samples,
            self.get_control_delta()
        )


class FLClientFedDC(FLClient):
    """
    FedDC Client (C-6): Federated Drift Correction.
    
    FedDC addresses data heterogeneity by maintaining local drift variables
    that capture the difference between local and global optima.
    
    Key idea: Each client maintains a drift variable h_i that approximates
    the difference between its local gradient and the global gradient.
    
    Local update with drift correction:
    w_i = w_global - eta * (g_i - h_i)
    
    Drift update after local training:
    h_i = h_i + alpha * (w_local - w_global) / eta
    
    Reference: FedDC algorithm for handling non-IID data in federated learning.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset: ClientDataset,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        local_epochs: int = 1,
        alpha: float = 0.1
    ):
        """
        Initialize FedDC client.
        
        Args:
            alpha: Drift update coefficient (default 0.1)
                   Controls how quickly the drift variable adapts.
                   Higher alpha = faster adaptation but potentially less stable.
        """
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            local_epochs=local_epochs
        )
        self.alpha = alpha
        
        # Drift variable h_i (initialized to zero)
        self.drift: Optional[OrderedDict] = None
        self.global_weights: Optional[OrderedDict] = None
    
    def receive_global_model(self, global_weights: OrderedDict, input_channels: int = 17):
        """Receive global model and store for drift correction."""
        super().receive_global_model(global_weights, input_channels)
        self.global_weights = copy.deepcopy(global_weights)
        
        # Initialize drift variable if needed
        if self.drift is None:
            self.drift = OrderedDict()
            for name, param in self.model.named_parameters():
                self.drift[name] = torch.zeros_like(param, device='cpu')
    
    def train_local(self) -> Dict[str, float]:
        """
        Perform local training with drift correction.
        
        The gradient is corrected by subtracting the drift variable:
        g_corrected = g - h_i
        
        This helps align local updates with the global objective.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call receive_global_model() first.")
        
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                
                # Apply drift correction to gradients
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in self.drift:
                            # g_corrected = g - h_i
                            h_i = self.drift[name].to(self.device)
                            param.grad.data.sub_(h_i)
                
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_y)
                epoch_samples += len(batch_y)
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # Update drift variable after training
        self._update_drift()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {'train_loss': avg_loss}
    
    def _update_drift(self):
        """
        Update drift variable after local training.
        
        h_i_new = h_i + alpha * (w_global - w_local) / (K * eta)
        
        where:
        - alpha: drift update coefficient
        - K: number of local steps
        - eta: learning rate
        
        This captures the "drift" between local and global models.
        """
        if self.global_weights is None:
            return
        
        # Estimate number of local steps
        K = self.local_epochs * len(self.train_loader)
        
        new_drift = OrderedDict()
        for name, param in self.model.named_parameters():
            h_i = self.drift[name].to(self.device)
            w_global = self.global_weights[name].to(self.device)
            w_local = param.data
            
            # h_i_new = h_i + alpha * (w_global - w_local) / (K * eta)
            delta = self.alpha * (w_global - w_local) / (K * self.learning_rate)
            new_drift[name] = (h_i + delta).cpu()
        
        self.drift = new_drift
    
    def get_drift(self) -> OrderedDict:
        """Get current drift variable."""
        return copy.deepcopy(self.drift) if self.drift else OrderedDict()
    
    def get_update(self) -> Tuple[int, OrderedDict, int, OrderedDict]:
        """
        Get client update tuple including drift variable.
        
        Returns:
            (client_id, state_dict, num_samples, drift)
        """
        return (
            self.client_id,
            self.get_model_weights(),
            self.num_samples,
            self.get_drift()
        )
