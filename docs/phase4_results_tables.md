## Phase 4 Results (TTA target = 40.0% val accuracy)

### Scenario 0 – Homogeneous

| Mode   | Final Acc (%) | TTA (s)  | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:--------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  44.5 ± 0.5   | 843 ± 33 | 1122 ± 43 |        22.8        |   0.0    |    0.0    |  100%  |
| async  |  45.9 ± 0.7   | 884 ± 16 | 1128 ± 28 |        45.2        |   0.0    |   714.2   |   0%   |
| hybrid |  44.0 ± 2.1   | 846 ± 25 | 1122 ± 34 |        22.8        |   0.0    |    0.0    |  100%  |

### Scenario A – Persistent Straggler

| Mode   | Final Acc (%) | TTA (s)  | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:--------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  43.2 ± 1.9   | 1409 ± 17 | 1878 ± 25 |        13.6        |   0.0    |    0.0    |  100%  |
| async  |  39.0 ± 0.6   | 641 ± 0  | 809 ± 42 |        43.2        |   0.0    |   373.5   |   0%   |
| hybrid |  45.2 ± 1.0   | 1402 ± 32 | 1865 ± 42 |        13.7        |   0.0    |    0.0    |  100%  |

### Scenario B – Transient Straggler

| Mode   | Final Acc (%) | TTA (s)  | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:--------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  46.3 ± 1.9   | 1323 ± 31 | 1631 ± 29 |        18.4        |   0.0    |    0.0    |  100%  |
| async  |  43.0 ± 0.7   | 760 ± 112 | 905 ± 17 |        45.3        |   0.0    |   458.1   |   0%   |
| hybrid |  47.1 ± 0.9   | 1178 ± 204 | 1602 ± 15 |        18.7        |   0.0    |    0.0    |  100%  |


## Phase 4 Mode Switch Events

| Run Tag | Global Step | Old Mode | New Mode | Trigger Reason | W-TimeVar | GradNormVar | LossTrend |
|---------|:-----------:|:--------:|:--------:|:-------------- |:---------:|:-----------:|:---------:|
| (no switch events found) | — | — | — | — | — | — | — |