_Sweep summary, figures, and limitations: [phase4_results_discussion.md](phase4_results_discussion.md)._

## Phase 4 Results (TTA target = 40.0% val accuracy)

### Scenario 0 – Homogeneous

| Mode   | Final Acc (%) | TTA (s)  | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:--------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  44.5 ± 0.5   | 843 ± 33 | 1122 ± 43 |        22.8        |   0.0    |    0.0    |  100%  |
| async  |  45.9 ± 0.7   | 884 ± 16 | 1128 ± 28 |        45.2        |   0.0    |   714.2   |   0%   |
| hybrid | — | — | — | — | — | — | — |

### Scenario A – Persistent Straggler

| Mode   | Final Acc (%) | TTA (s)  | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:--------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  43.2 ± 1.9   | 1409 ± 17 | 1878 ± 25 |        13.6        |   0.0    |    0.0    |  100%  |
| async  |  39.0 ± 0.6   | 641 ± 0  | 809 ± 42 |        43.2        |   0.0    |   373.5   |   0%   |
| hybrid |  41.9 ± 2.9   | 1233 ± 386 | 1609 ± 95 |        23.3        |   1.7    |   359.6   |  20%   |

### Scenario B – Transient Straggler

| Mode   | Final Acc (%) | TTA (s)  | Wall (s) | Throughput (img/s) | Switches | Staleness | Sync % |
|--------|:-------------:|:--------:|:--------:|:------------------:|:--------:|:---------:|:------:|
| sync   |  46.3 ± 1.9   | 1323 ± 31 | 1631 ± 29 |        18.4        |   0.0    |    0.0    |  100%  |
| async  |  43.0 ± 0.7   | 760 ± 112 | 905 ± 17 |        45.3        |   0.0    |   458.1   |   0%   |
| hybrid |  36.9 ± 3.6   | 864 ± 0  | 1160 ± 116 |        33.5        |   2.3    |   331.0   |  22%   |


## Phase 4 Mode Switch Events

| Run Tag | Global Step | Old Mode | New Mode | Trigger Reason | W-TimeVar | GradNormVar | LossTrend |
|---------|:-----------:|:--------:|:--------:|:-------------- |:---------:|:-----------:|:---------:|
| p4_sA_hybrid_s123 | 120 | sync | async | straggler_detected | 0.4053 | 0.0265 | -0.0168 |
| p4_sA_hybrid_s123 | 360 | async | sync | loss_rising | 0.3793 | 0.0337 | 0.0726 |
| p4_sA_hybrid_s123 | 480 | sync | async | straggler_detected | 0.4106 | 0.0192 | 0.0192 |
| p4_sA_hybrid_s42 | 120 | sync | async | straggler_detected | 0.3735 | 0.0184 | -0.0137 |
| p4_sA_hybrid_s777 | 120 | sync | async | straggler_detected | 0.3886 | 0.0192 | -0.0269 |
| p4_sB_hybrid_s123 | 120 | sync | async | straggler_detected | 0.4426 | 0.0289 | -0.0162 |
| p4_sB_hybrid_s123 | 760 | async | sync | loss_rising | 0.5091 | 0.0469 | 0.0702 |
| p4_sB_hybrid_s123 | 880 | sync | async | straggler_detected | 0.5278 | 0.0292 | -0.0464 |
| p4_sB_hybrid_s42 | 120 | sync | async | straggler_detected | 0.4309 | 0.0196 | -0.0262 |
| p4_sB_hybrid_s42 | 320 | async | sync | loss_rising | 0.4687 | 0.0492 | 0.1210 |
| p4_sB_hybrid_s42 | 440 | sync | async | straggler_detected | 0.4981 | 0.0102 | 0.0180 |
| p4_sB_hybrid_s777 | 120 | sync | async | straggler_detected | 0.4510 | 0.0179 | -0.0192 |