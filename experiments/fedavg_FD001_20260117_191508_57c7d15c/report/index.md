# FEDAVG Experiment Report — FD001

**Experiment ID:** `fedavg_FD001_20260117_191508_57c7d15c`  
**Generated:** 2026-01-17 20:58:11  
**Author:** FL Experiment Runner

---

## 1. Configuration Summary

| Parameter | Value |
|-----------|-------|
| Algorithm | fedavg |
| Dataset | FD001 |
| Rounds | 100 |
| Local Epochs | 1 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Participation | 100% |
| Num Clients | 2 |
| Seed | 42 |

---

## 2. Final Results

| Metric | Value |
|--------|-------|
| **Test RMSE** | **30.07** |
| **Test MAE** | **28.67** |
| Total Time | 23.8s |
| Avg Round Time | 0.24s |

---

## 3. Convergence Analysis

Training completed **200** rounds.

- **Initial RMSE:** 36.95
- **Final RMSE:** 7.33
- **Best RMSE:** 3.42 (Round 134)
- **Improvement:** 33.53 (90.7%)

### Plots

![client_variance](../plots/client_variance.png)

![convergence_rmse](../plots/convergence_rmse.png)

![rmse_mae](../plots/rmse_mae.png)

---

## 4. Data Distribution (Non-IID)

| Statistic | Value |
|-----------|-------|
| Num Clients | 2 |
| Total Samples | 294 |
| Min Samples/Client | 114 |
| Max Samples/Client | 180 |
| Mean Samples/Client | 147.0 |
| Std Samples/Client | 46.7 |
| Mean RUL Range | [99.3, 118.7] |

---

## 5. Files & Artifacts

### Logs
- [rounds.csv](../logs/rounds.csv) — Round-wise metrics
- [non_iid.csv](../logs/non_iid.csv) — Per-client data statistics
- [summary.json](../logs/summary.json) — Final experiment summary

### Model
- [final_model.pt](../final_model.pt) — Final global model checkpoint

### Configuration
- [config.json](../config.json) — Full experiment configuration

---

*Report generated automatically by FL Experiment Runner*
