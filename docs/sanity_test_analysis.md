# S-AN-2: Aggregation Discrepancy Analysis

**Date:** 2026-01-17  
**Phase:** 2.5 — Algorithm Sanity Tests

## Summary

Phase 2.5 sanity tests identified and documented the expected differences between FedAvg and centralized training. **No bugs found** — the observed differences are mathematically expected behavior.

## Findings

### 1. FedAvg ≠ Centralized SGD (Expected)

With IID data, 3 clients, and 1 round:
- **Centralized loss reduction:** ~15-25%
- **FedAvg loss reduction:** ~15-25%  
- **Weight change correlation:** >0.5 (same direction)
- **Max weight difference:** ~3.0 (expected, not a bug)

### 2. Root Causes of Difference

| Factor | Impact | Notes |
|--------|--------|-------|
| Independent Adam states | High | Each client maintains separate momentum/variance |
| Weight averaging | High | FedAvg averages MODEL weights, not gradients |
| Batch ordering | Medium | Different shuffle per client |
| Learning rate decay | N/A | Not used (fixed lr=1e-3) |

### 3. Mathematical Explanation

**Centralized SGD with Adam:**
```
θ_{t+1} = θ_t - lr * m_t / (√v_t + ε)
```
where `m_t`, `v_t` accumulate across ALL data.

**FedAvg with Adam (per client):**
```
θ_k^{t+1} = θ_t - lr * m_k^t / (√v_k^t + ε)  // per-client
θ_{t+1} = Σ (n_k / N) * θ_k^{t+1}            // aggregate weights
```
Each client has LOCAL `m_k`, `v_k` that don't transfer.

**Key insight:** FedAvg averages the destination (weights after training), not the journey (optimizer states). This is fundamentally different from centralized training.

## Tests Implemented

| Test | Description | Status |
|------|-------------|--------|
| `test_iid_fedavg_vs_centralized_same_direction` | Verify both methods reduce loss and correlate | ✅ PASS |
| `test_fedavg_deterministic_with_seed` | Same seed → identical results | ✅ PASS |
| `test_fedavg_aggregation_weights_correct` | Weighted avg math is correct | ✅ PASS |
| `test_fedavg_improves_loss` | Training actually helps | ✅ PASS |
| `test_equal_samples_equal_weight` | Equal samples → equal influence | ✅ PASS |
| `test_single_batch_gradient_match` | Single batch ≈ closer match | ✅ PASS |
| `test_no_nan_in_aggregation` | No numerical issues | ✅ PASS |
| `test_zero_sample_client_excluded` | Edge case handling | ✅ PASS |
| `test_large_sample_imbalance` | Extreme imbalance works | ✅ PASS |
| `test_weight_dtype_preservation` | float32 preserved | ✅ PASS |

## Recommendations

1. **No fixes required** — Current FedAvg implementation is correct
2. **For tighter approximation** (if desired):
   - Use SGD instead of Adam (removes per-client optimizer state)
   - Use FedSGD (aggregate gradients, not weights)
   - These are algorithm changes, not bug fixes

3. **Validation for Phase 3:**
   - FedProx: Add proximal term to local objective
   - SCAFFOLD: Track control variates to correct client drift
   - FedDC: Daisy-chain corrections

## Conclusion

Phase 2.5 sanity tests **PASSED**. The FedAvg implementation:
- ✅ Correctly aggregates weights by sample count
- ✅ Deterministically reproducible with seeds
- ✅ Reduces loss on IID data
- ✅ Moves weights in same direction as centralized
- ✅ Handles edge cases (imbalance, dtypes)

The observed difference from centralized training is **expected mathematical behavior**, not a bug.
