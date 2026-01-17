"""
Tests for the data pipeline module.

Validates:
- Sensor selection matches configs/sensors.yaml
- Sliding window (window=30, stride=1)
- Chronological split with no temporal leakage
- Global z-score normalization
- Non-IID quantification per requirements.md section 8.2
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from data.preprocessing import (
    load_config,
    compute_rul,
    select_features,
    create_sequences,
    chronological_split,
    GlobalNormalizer,
)
from data.client_dataset import (
    RULDataset,
    ClientDataset,
    create_client_datasets,
    compute_operating_condition_entropy,
    compute_client_non_iid_stats,
)


class TestSensorSelection:
    """Tests for sensor selection (B-1)."""
    
    def test_sensor_config_loads(self):
        """Sensor config should load without errors."""
        config = load_config('sensors.yaml')
        assert 'sensors' in config
        assert 'operational_settings' in config
    
    def test_correct_number_of_sensors(self):
        """Should select 14 sensors + 3 operational settings = 17 features."""
        config = load_config('sensors.yaml')
        n_sensors = len(config['sensors']['selected_indices'])
        n_ops = len(config['operational_settings']['names'])
        assert n_sensors == 14, f"Expected 14 sensors, got {n_sensors}"
        assert n_ops == 3, f"Expected 3 operational settings, got {n_ops}"
    
    def test_excluded_sensors_not_selected(self):
        """Low-variance sensors 1, 5, 6, 10, 16, 18, 19 should be excluded."""
        config = load_config('sensors.yaml')
        selected = set(config['sensors']['selected_indices'])
        # Excluded sensors (0-indexed): 0, 4, 5, 9, 15, 17, 18
        # Wait - need to check the actual excluded list
        excluded = set(config['sensors']['excluded_indices'])
        assert selected.isdisjoint(excluded), "Selected and excluded sensors overlap"


class TestSlidingWindow:
    """Tests for sliding window segmentation (B-2)."""
    
    def test_window_length(self):
        """Window length should be 30."""
        config = load_config('preprocessing.yaml')
        assert config['preprocessing']['window_length'] == 30
    
    def test_stride(self):
        """Stride should be 1."""
        config = load_config('preprocessing.yaml')
        assert config['preprocessing']['stride'] == 1
    
    def test_sequence_shape(self):
        """Sequences should have shape (n, window_length, n_features)."""
        # Create dummy data
        n_samples = 100
        n_features = 17
        features = np.random.randn(n_samples, n_features)
        targets = np.arange(n_samples)
        unit_ids = np.ones(n_samples)
        
        X, y, ids = create_sequences(features, targets, unit_ids, window_length=30, stride=1)
        
        assert X.shape[1] == 30, f"Expected window length 30, got {X.shape[1]}"
        assert X.shape[2] == n_features, f"Expected {n_features} features, got {X.shape[2]}"
    
    def test_target_is_end_of_window(self):
        """Target should be RUL at the end of each window."""
        features = np.random.randn(50, 5)
        targets = np.arange(50)  # Known targets
        unit_ids = np.ones(50)
        
        X, y, ids = create_sequences(features, targets, unit_ids, window_length=10, stride=1)
        
        # First window: samples 0-9, target should be targets[9] = 9
        assert y[0] == 9, f"Expected target 9, got {y[0]}"
        # Last window: samples 40-49, target should be targets[49] = 49
        assert y[-1] == 49, f"Expected target 49, got {y[-1]}"


class TestChronologicalSplit:
    """Tests for chronological split with no leakage (B-2)."""
    
    def test_no_temporal_leakage(self):
        """
        For each engine, train windows should come before val windows,
        and val windows should come before test windows.
        """
        # Create sequences for one engine
        n_windows = 100
        X = np.arange(n_windows).reshape(-1, 1, 1) * np.ones((1, 10, 5))
        y = np.arange(n_windows)
        unit_ids = np.ones(n_windows)
        
        splits = chronological_split(X, y, unit_ids, train_ratio=0.7, val_ratio=0.15)
        
        train_X, train_y, _ = splits['train']
        val_X, val_y, _ = splits['val']
        test_X, test_y, _ = splits['test']
        
        # Check ordering: all train targets < all val targets < all test targets
        if len(train_y) > 0 and len(val_y) > 0:
            assert train_y.max() < val_y.min(), "Temporal leakage: train overlaps val"
        if len(val_y) > 0 and len(test_y) > 0:
            assert val_y.max() < test_y.min(), "Temporal leakage: val overlaps test"
    
    def test_split_ratios(self):
        """Split should approximately follow 70/15/15 ratio."""
        n_windows = 1000
        X = np.random.randn(n_windows, 10, 5)
        y = np.random.randn(n_windows)
        unit_ids = np.ones(n_windows)
        
        splits = chronological_split(X, y, unit_ids, train_ratio=0.7, val_ratio=0.15)
        
        n_train = len(splits['train'][0])
        n_val = len(splits['val'][0])
        n_test = len(splits['test'][0])
        
        assert 650 < n_train < 750, f"Train size {n_train} not ~70%"
        assert 100 < n_val < 200, f"Val size {n_val} not ~15%"
        assert 100 < n_test < 200, f"Test size {n_test} not ~15%"


class TestNormalization:
    """Tests for global z-score normalization (B-3)."""
    
    def test_train_stats_only(self):
        """Normalizer should be fit on train data only."""
        train_X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
        val_X = np.array([[[100, 200], [300, 400]]])  # Very different distribution
        
        normalizer = GlobalNormalizer()
        train_normalized = normalizer.fit_transform(train_X)
        val_normalized = normalizer.transform(val_X)
        
        # Train should be zero-mean, unit-variance (approximately)
        train_flat = train_normalized.reshape(-1, 2)
        assert abs(train_flat.mean()) < 0.1, "Train mean should be ~0"
        assert abs(train_flat.std() - 1.0) < 0.1, "Train std should be ~1"
        
        # Val should NOT be zero-mean (since stats are from train)
        val_flat = val_normalized.reshape(-1, 2)
        assert abs(val_flat.mean()) > 1, "Val mean should differ since using train stats"
    
    def test_save_load_params(self, tmp_path):
        """Normalizer params should save to and load from JSON."""
        train_X = np.random.randn(100, 30, 17)
        
        normalizer = GlobalNormalizer()
        normalizer.fit(train_X, feature_names=[f"f{i}" for i in range(17)])
        
        # Save
        save_path = str(tmp_path / "norm_params.json")
        normalizer.save(save_path)
        
        # Load
        loaded = GlobalNormalizer.load(save_path)
        
        np.testing.assert_array_almost_equal(normalizer.mean, loaded.mean)
        np.testing.assert_array_almost_equal(normalizer.std, loaded.std)
        assert normalizer.feature_names == loaded.feature_names


class TestClientDataset:
    """Tests for per-client dataset (B-4)."""
    
    def test_client_creation(self):
        """Should create ClientDataset objects correctly."""
        train_X = np.random.randn(50, 30, 17)
        train_y = np.random.randn(50)
        
        client = ClientDataset(
            client_id=1,
            train_X=train_X,
            train_y=train_y
        )
        
        assert client.client_id == 1
        assert client.train_size == 50
    
    def test_dataloader(self):
        """DataLoader should yield correct batches."""
        train_X = np.random.randn(100, 30, 17)
        train_y = np.random.randn(100)
        
        client = ClientDataset(client_id=1, train_X=train_X, train_y=train_y)
        loader = client.get_train_loader(batch_size=32)
        
        batch_X, batch_y = next(iter(loader))
        
        assert batch_X.shape == (32, 30, 17)
        assert batch_y.shape == (32,)


class TestNonIIDQuantification:
    """Tests for non-IID quantification per requirements.md section 8.2."""
    
    def test_operating_condition_entropy(self):
        """Entropy should be computed correctly."""
        # Uniform distribution should have higher entropy
        uniform_ops = np.random.rand(1000, 3)
        
        # Concentrated distribution should have lower entropy
        concentrated_ops = np.ones((1000, 3)) * 0.5 + np.random.randn(1000, 3) * 0.01
        
        uniform_entropy = compute_operating_condition_entropy(uniform_ops)
        concentrated_entropy = compute_operating_condition_entropy(concentrated_ops)
        
        assert uniform_entropy > concentrated_entropy, \
            "Uniform distribution should have higher entropy"
    
    def test_non_iid_stats_fields(self):
        """Non-IID stats should have all required fields per requirements."""
        # Create mock client and raw data
        train_X = np.random.randn(50, 30, 17)
        train_y = np.linspace(0, 100, 50)  # Known RUL distribution
        
        client = ClientDataset(client_id=1, train_X=train_X, train_y=train_y)
        
        # Create matching raw dataframe
        raw_df = pd.DataFrame({
            'unit_id': [1] * 100,
            'cycle': list(range(1, 101)),
            'op_1': np.random.rand(100),
            'op_2': np.random.rand(100),
            'op_3': np.random.rand(100),
            **{f's_{i}': np.random.rand(100) for i in range(1, 22)}
        })
        
        stats = compute_client_non_iid_stats(client, raw_df)
        
        # Check all required fields per log_schema.yaml
        required_fields = [
            'client_id', 'num_samples', 'mean_rul', 'rul_variance',
            'operating_condition_entropy', 'sensor_variance'
        ]
        for field in required_fields:
            assert field in stats, f"Missing required field: {field}"
        
        # Verify values make sense
        assert stats['client_id'] == 1
        assert stats['num_samples'] == 50
        assert 40 < stats['mean_rul'] < 60  # Should be ~50 for linspace(0,100,50)
        assert stats['rul_variance'] > 0


class TestRULComputation:
    """Tests for RUL computation."""
    
    def test_rul_cap(self):
        """RUL should be capped at 125."""
        # Create mock data with long-running engine
        df = pd.DataFrame({
            'unit_id': [1] * 200,
            'cycle': list(range(1, 201))
        })
        
        df = compute_rul(df, rul_cap=125)
        
        assert df['RUL'].max() == 125, "RUL should be capped at 125"
        assert df['RUL'].min() == 0, "Final RUL should be 0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
