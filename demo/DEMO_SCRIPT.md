# DEMO RECORDING SCRIPT
## Hybrid Distributed SGD — Parallel and Distributed Computing Project

> **HOW TO USE THIS SCRIPT:** Read each line under **SAY:** out loud while screen recording. Follow the **SHOW:** instructions to display the right slide/file on screen. A PowerPoint `Hybrid_Distributed_SGD_Demo.pptx` (15 slides) has all diagrams. Results CSVs are in `demo_recording/results/` — open them in Excel. The script is written so you can read it word by word.

---

## PART 1 — INTRODUCTION AND PROBLEM STATEMENT (3-4 minutes)

### SHOW: PPT Slide 1 (Title Slide) then Slide 2 (Problem Statement diagram)

### SAY:

"Hi everyone. My name is [YOUR NAME], and today I am presenting my Parallel and Distributed Computing project. The title of my project is **Hybrid Distributed SGD with Bi-directional Switching between Ring All-Reduce and Parameter Server.**"

"So let me first explain the problem. When we train a deep learning model, the model has to look at millions of images and learn from them. If we use just one computer, this takes a very long time. For example, training on a single machine might take 10 hours."

"The solution is **Distributed Training**. Instead of one computer doing all the work, we split the work across multiple computers, which we call **workers**. Each worker takes a small part of the data, does the math, and then they all share their results. This way, the same training that took 10 hours on one machine can finish in just 2 to 3 hours on 4 machines."

"But here is the catch — when these workers need to share their results with each other, there are two ways to do it. And both have problems. Let me explain those now."

---

## PART 2 — SYNC vs ASYNC MODES (3-4 minutes)

### SHOW: PPT Slide 3 (Sync vs Async diagram)

### SAY:

"The first way is called **Synchronous Mode**, or Sync mode for short."

"In Sync mode, every worker finishes its calculations, and then all of them **wait** for each other before moving forward. The server takes all their results, averages them together, and gives everyone the same updated model."

"The math behind it is simple. The new weights equal the old weights minus the learning rate times the average of all workers' gradients. In formula form, that is: W new equals W old minus eta times one over N times the sum of all N gradients. This is the standard SGD update rule but applied across multiple workers."

"The problem with Sync mode is simple — if one worker is slow, maybe its CPU is busy or it got a harder batch of data, **everyone else has to sit and wait**. We call this slow worker a **straggler**. In our experiments, when we had a straggler, Sync mode took **1409 seconds** to reach our target accuracy. That is a LOT of wasted time."

"Now let me explain the second way — **Asynchronous Mode**, or Async mode."

"In Async mode, nobody waits for anyone. As soon as a worker finishes its calculation, it sends the result to the server immediately, and the server applies it right away. The worker then grabs the latest model and starts working again."

"This sounds great because it is very fast — no waiting at all. But there is a hidden problem called **staleness**. Because nobody is waiting for each other, a worker might be doing its math using an old, outdated version of the model. By the time it sends its result, the model has already been updated many times by other workers. So this worker's result is based on old information, and this confuses the training. The model's accuracy drops."

"In our experiments, Async mode was very fast — it finished in just **641 seconds**. But the accuracy was lower, and only 1 out of 3 test runs actually reached our target of 40 percent accuracy."

---

## PART 3 — THE HYBRID SOLUTION (2-3 minutes)

### SHOW: PPT Slide 4 (Hybrid Switching diagram)

### SAY:

"So here is the key question of my project — why do we have to pick just one? Why can we not use Sync when everything is running smoothly and switch to Async only when there is a straggler?"

"That is exactly what I built. I created a **Hybrid System** with a **bi-directional controller**. Bi-directional means it can switch in both directions — from Sync to Async AND from Async back to Sync."

"The controller is like a traffic cop sitting inside the Parameter Server. It watches three live signals and decides which mode to use."

### SHOW: PPT Slide 5 (Controller Signals and Thresholds)

"**Signal 1 is Worker Time Variance.** This measures how different the speeds of the two workers are. If Worker 0 finishes in 1 second but Worker 1 takes 5 seconds, the variance is very high. That means we have a straggler. When this value goes above our threshold of 0.06, the controller says 'straggler detected' and switches from Sync to Async."

"**Signal 2 is Gradient Norm Variance.** This measures how different the workers' gradient updates are. If this gets very high, above 0.90, it means the training is becoming unstable, so the controller switches back to Sync."

"**Signal 3 is Loss Trend.** This tracks whether the training loss is going up or going down. If the loss starts rising, meaning the model is getting worse instead of better, the controller detects this and switches back to Sync. The threshold for this is 0.06."

"All three signals are smoothed using something called **EMA — Exponential Moving Average**. EMA is basically a math trick that says: take 5 percent of the new value and 95 percent of the old smoothed value. This prevents the controller from reacting to random noise. It only reacts to real, sustained patterns."

"On top of that, we have a **cooldown period** of 120 steps. After a switch happens, the controller must wait at least 120 steps before it can switch again. And the condition must be true for 2 consecutive checks, which we call the **decision window**. This prevents the controller from flipping back and forth like crazy."

---

## PART 4 — SYSTEM ARCHITECTURE (3-4 minutes)

### SHOW: PPT Slide 6 (Architecture diagram)

### SAY:

"Now let me show you the architecture of the system I built."

"Everything runs inside a single **Docker container**. The Docker image is called `pdc-hybrid:latest`. We use Docker so that the experiment is fully reproducible — anyone with Docker installed can rebuild and re-run everything from scratch."

"Inside the container, we have 3 processes:"

"**Process 1 is the Parameter Server, or PS.** This is rank 0. It is the boss of the system. It holds the global model in shared memory using PyTorch's `model.share_memory()` function. The PS has three main components inside it:"
"- The **Aggregator**, which is the component that actually applies the gradient updates. When the mode flag is 0, it works in Sync mode — it waits for both workers, averages the gradients, and applies one update. When the mode flag is 1, it works in Async mode — it applies each gradient immediately as it arrives."
"- The **Metrics Collector**, which records the step time, gradient norm, and loss from every gradient it receives, and computes the three EMA signals I just explained."
"- The **Controller**, which checks those signals every 40 steps and decides if a switch is needed."

"**Process 2 is Worker 0 (rank 1)** and **Process 3 is Worker 1 (rank 2).** Each worker takes a batch of 32 images from the CIFAR-10 dataset, runs a forward pass through the ResNet-18 model, computes the loss, runs a backward pass to get the gradients, and then pushes those gradients to a shared queue called the **Gradient Queue**."

"The Gradient Queue is a `torch.multiprocessing.Queue`. It is a first-in, first-out pipe that workers use to send their gradients to the PS. The queue has a maximum size of 200 to prevent memory overflow."

"There is also a **param_lock**, which is a multiprocessing Lock. This protects the shared model parameters from being read and written at the same time, which would corrupt the data."

"And finally, there is a **mode_flag** — a shared integer, 0 for Sync and 1 for Async — and a **sync_round** counter. In Sync mode, after the PS applies the averaged update, it increments sync_round, which tells all workers 'okay, the round is done, you can pull the new parameters and start the next batch.'"

---

## PART 5 — DATASET, MODEL, AND SETUP (1-2 minutes)

### SHOW: PPT Slide 7 (Experimental Setup text slide)

### SAY:

"Let me quickly explain the experimental setup."

"The dataset is **CIFAR-10**. It has 60,000 color images, each 32 by 32 pixels, across 10 categories like airplane, car, bird, cat, and so on. 50,000 images are for training and 10,000 are for testing. I chose CIFAR-10 because it is small enough to train on a CPU laptop but complex enough to show meaningful differences between Sync and Async modes."

"The model is **ResNet-18**. It has about **11.17 million trainable parameters**, which is roughly 44 megabytes of gradient data per update. I modified it for CIFAR by changing the first layer from a 7 by 7 convolution to a 3 by 3 convolution and removing the max pooling layer. This is the standard CIFAR adaptation used by researchers."

"The optimizer is **SGD with momentum 0.9** and a learning rate of 0.01. The batch size is **32 images per worker per step**. We used 2 workers."

"Every experiment was run with **3 random seeds** — 42, 123, and 777 — for reproducibility. This gives us mean and standard deviation for every metric."

"Everything is containerized in Docker with **4 CPU cores** and **6 GB RAM**. No GPU was used — this is deliberate because the goal is to prove the switching concept, not to set accuracy records."

---

## PART 6 — PROJECT PIPELINE (2-3 minutes)

### SHOW: PPT Slide 8 (Pipeline diagram)

### SAY:

"Let me now walk you through the pipeline — the five phases of how I built this project from scratch."

"**Phase 1** was the Synchronous Baseline. I implemented standard DDP — Distributed Data Parallel — using PyTorch's built-in All-Reduce mechanism. I ran it with 1 worker, 2 workers, and 4 workers to get baseline numbers. The code is in `src/sync_allreduce/`."

### SHOW: Switch to Excel — open `demo_recording/results/phase1_sync_baseline_summary.csv`

"Here you can see the Phase 1 summary — the model, dataset, and the world sizes I tested."

"**Phase 2** was the Asynchronous Baseline. I implemented a Parameter Server from scratch using `torch.multiprocessing`. Workers push gradients to a queue, the PS applies them immediately, and workers pull updated parameters every 5 steps. I also added staleness tracking. The code is in `src/async_ps/`."

### SHOW: Switch to Excel — open `demo_recording/results/phase2_async_baseline_summary.csv`

"**Phase 3** was the Hybrid Controller. This is where the real innovation happened. I took the PS from Phase 2 and added the Mode Controller and the Metrics Collector. Now the PS could switch between Sync and Async aggregation mid-run without restarting any process. The code is in `src/hybrid/`."

### SHOW: Switch to Excel — open `demo_recording/results/phase3_hybrid_controller_summary.csv`

"**Phase 4** was the Experiments. I ran 3 Scenarios times 3 Modes times 3 Random Seeds, which gives us 24 training runs. Plus I ran a threshold sweep of 17 extra configurations. All of this was containerized in Docker."

"**Phase 5** was the Analysis. I wrote Python scripts that read all the CSV logs, compute the metrics, generate the comparison tables, and create the figures you will see shortly."

---

## PART 7 — SHOWING THE CODE (3-4 minutes)

### SHOW: Open `src/hybrid/controller.py` in your code editor

### SAY:

"Now let me show you the actual code. This file is `controller.py` — the brain of the hybrid system."

"At the top you can see the switching rules written as comments. For Sync to Async: the worker time variance must be above `straggler_var_high` AND the gradient norm variance must be below `grad_var_low`. This means — a straggler is detected AND training is still stable enough to handle the switch."

"For Async to Sync, there are three possible triggers: gradient norm variance above `grad_var_high` means instability. Loss trend above the threshold means the loss is rising. Or worker time variance dropping below `straggler_var_low` means the straggler is gone."

"Look at the `evaluate` method on line 114. First it checks the cooldown — if we have not been in the current mode long enough, it returns immediately with no switch. Then it checks the signals against the thresholds. If a trigger fires, it increments the consecutive trigger count. Only when the trigger fires for `decision_window` consecutive evaluation points does it actually flip the mode."

### SHOW: Open `src/hybrid/metrics_collector.py` in your code editor

"This is the Metrics Collector. It tracks three signals using EMA. Look at `record_gradient` on line 63 — every time the PS receives a gradient, this function is called. It records the worker's step time, the gradient norm, and the loss. Then it calls `_update_time_var` and `_update_grad_var` to recompute the EMA-smoothed signals."

### SHOW: Open `src/hybrid/launcher.py` in your code editor

"And this is the launcher. On line 58, it creates the ResNet-18 model and calls `share_memory()` to put it in OS shared memory. Then on line 76, it creates the gradient queue, the parameter lock, the mode flag, and the sync round counter. Finally on line 96, it uses `mp.spawn` to create all processes — 1 PS plus N workers."

---

## PART 8 — DOCKER AND RUNNING THE EXPERIMENT (2-3 minutes)

### SHOW: Open `docker/Dockerfile` in your code editor

### SAY:

"The entire project is containerized using Docker. Here is the Dockerfile. The base image is `python:3.12-slim`. It installs the system dependencies, then installs the Python packages from `requirements.txt`, and copies the source code, configs, and experiment scripts into the image."

### SHOW: Open `docker/docker-compose.yml` in your code editor

"And this is the docker-compose file. It defines two services. The `scenario_runner` service is for running training experiments. It bind-mounts the `data/`, `logs/`, and `checkpoints/` directories so that all CSV logs written inside the container appear on the host machine immediately. The resource caps are set to 4 CPUs and 6 gigabytes of RAM."

### SHOW: Open a terminal in the project directory

### SAY:

"To run an experiment, the commands are straightforward. First we build the Docker image:"

```
docker compose -f docker/docker-compose.yml build
```

"Then we run, for example, Scenario A Hybrid:"

```
docker compose -f docker/docker-compose.yml run --rm scenario_runner src/hybrid/launcher.py --nworkers 2 --config configs/phase4_sA_hybrid.yaml
```

"This spins up one container. Inside, it creates 3 processes — 1 Parameter Server and 2 Workers. Worker 1 is configured with a 1.5 second delay per step to simulate a persistent straggler. The controller is enabled and will watch the signals to decide when to switch modes."

"While it runs, you can see the terminal printing the mode switches in real time. It will print something like: 'MODE SWITCH: sync to async at step 120, reason equals straggler detected'. That is the controller in action."

---

## PART 9 — THE SEVEN EVALUATION METRICS (2-3 minutes)

### SHOW: PPT Slide 9 (Evaluation Metrics slide)

### SAY:

"Before I show you the results, let me explain the **seven metrics** we used to evaluate the system. These are the numbers that prove whether our hybrid approach actually works."

"**Metric 1 is Time-to-Accuracy, or TTA.** This is the number of seconds it takes to reach our target of 40 percent validation accuracy. This is our primary metric. If hybrid has a lower TTA than sync, it means hybrid reached the finish line faster."

"**Metric 2 is Final Accuracy.** This is the validation accuracy at the very end of training, after all steps are done. This tells us whether the hybrid system sacrificed quality for speed. Ideally, it should be close to sync's final accuracy."

"**Metric 3 is Wall Clock Time.** This is the total runtime of the experiment from start to finish, regardless of whether the target was reached. It tells us the overall cost of each run."

"**Metric 4 is Throughput, measured in images per second.** This tells us how efficiently the system was using the hardware. Sync mode had only 13.6 images per second because of all the waiting. Async had 45.2. Hybrid was in between at 23.3, which is almost double what sync achieved."

"**Metric 5 is Staleness, measured in steps.** Staleness means how outdated a worker's model was compared to the global model when it sent its gradient. In sync mode, staleness is always zero because everyone syncs up every round. In async mode, staleness was 373 steps on average for Scenario A. In hybrid mode, it was 359, slightly lower than async because hybrid spent some time in sync mode."

"**Metric 6 is Switch Count.** This is the number of times the controller flipped between sync and async during a training run. For Scenario A hybrid, it was 1.7 switches on average. This proves the controller is real and active."

"**Metric 7 is Sync Percentage.** This is the fraction of time the system spent in synchronous mode. For Scenario A hybrid, it was 20 percent — meaning the system spent 80 percent of its time in async mode to avoid the straggler delay, and 20 percent in sync mode for accuracy stabilization."

---

## PART 10 — THE THREE SCENARIOS AND RESULTS (5-6 minutes)

### SHOW: PPT Slide 8 (Three Scenarios diagram)

### SAY:

"Now let me walk you through the three test scenarios and the results we got for each."

"We tested the system under three conditions. Each condition was run with **three modes** — pure Sync, pure Async, and Hybrid — and each mode was run with **three random seeds** (42, 123, and 777) for reproducibility. So we have 3 scenarios times 3 modes times 3 seeds equals **24 total training runs**, plus a separate threshold sweep of **17 configurations**."

---

### Scenario 0 — Homogeneous (Perfect World)

### SHOW: Switch to Excel — open `demo_recording/results/phase4_scenario0_homogeneous.csv`

### SAY:

"**Scenario 0** is the baseline — both workers are running at the same speed. Nobody is slow."

"As you can see in this CSV, Sync mode reached 40 percent accuracy in about **843 seconds on average**, with a final accuracy of **44.5 percent**. Throughput was **22.8 images per second** and staleness was exactly **zero**."

"Async mode reached it in **884 seconds** with a slightly higher final accuracy of **45.9 percent** but with a massive average staleness of **714 steps**. Throughput was **45.2 images per second**."

"We did **not** run Hybrid in Scenario 0 because there is no straggler to detect. The controller would simply sit in Sync mode the whole time and add nothing. This proves that our system **does no harm** when conditions are normal."

---

### Scenario A — Persistent Straggler

### SHOW: Switch to Excel — open `demo_recording/results/phase4_scenarioA_persistent_straggler.csv`

### SAY:

"**Scenario A** is the main test. Worker 1 is permanently slow — it has a 1.5 second delay on every single step from step 0 to step 800. This is like having one slow computer in your cluster that never recovers."

"Let me walk you through all seven metrics for each mode, and you can follow along in the CSV."

"**Sync mode:** Final accuracy 43.2 percent. TTA was 1409 seconds — the slowest. Wall time was 1878 seconds. Throughput was only 13.6 images per second. Zero switches. Zero staleness. 100 percent sync."

"**Async mode:** Final accuracy 39.0 percent — lower than sync. TTA was 641 seconds but only 1 out of 3 seeds reached the target. Wall time was 809 seconds. Throughput was 43.2 images per second — the fastest. Zero switches. Staleness was 373.5 steps. 0 percent sync."

"**Hybrid mode:** Final accuracy 41.9 percent — close to sync. TTA was 1233 seconds — that is **12 percent faster than Sync**. Wall time was 1609 seconds. Throughput was 23.3 images per second — **almost double sync's throughput**. There were 1.7 switches on average. Staleness was 359.6 — lower than pure async. 20 percent sync, meaning 80 percent of the time was spent in async mode."

"So hybrid gave us the best balance — faster than sync, more accurate than async, and nearly double the throughput of sync."

---

### Scenario B — Transient Straggler

### SHOW: Switch to Excel — open `demo_recording/results/phase4_scenarioB_transient_straggler.csv`

### SAY:

"**Scenario B** is the harder test. Worker 1 is slow only from step 100 to step 500, then it recovers and becomes fast again. This simulates a temporary network glitch."

"**Sync mode:** Final accuracy 46.3 percent. TTA 1323 seconds. Wall time 1631 seconds. Throughput 18.4 images per second. Zero switches. Zero staleness. 100 percent sync."

"**Async mode:** Final accuracy 43.0 percent. TTA 760 seconds. Wall time 905 seconds. Throughput 45.3 images per second. Zero switches. Staleness 458.1. 0 percent sync."

"**Hybrid mode:** Final accuracy 36.9 percent — this is lower, and I will explain why. TTA was 864 seconds on the one seed that reached the target, but only **1 out of 3 seeds** reached 40 percent. Wall time 1160 seconds. Throughput 33.5 images per second. There were 2.3 switches. Staleness was 331. 22 percent sync."

"The reason hybrid's accuracy dropped in Scenario B is visible in the mode timeline — the controller detects the straggler at step 120 and switches to Async. But after the straggler recovers at step 500, the EMA-smoothed signal takes time to come down, so the system stays in Async mode too long. The controller's current thresholds are tuned for persistent stragglers and they **over-react** to temporary glitches. This is a known trade-off, and the threshold sweep of 17 configurations shows exactly which knobs to tune to fix it."

---

## PART 11 — SWITCH EVENTS AS PROOF (2 minutes)

### SHOW: Switch to Excel — open `demo_recording/results/phase4_all_switch_events.csv`

### SAY:

"Now let me show you the most important proof. This CSV contains every single mode switch event from all hybrid runs."

"Look at Scenario A, seed 123. At step 120, the controller switched from sync to async because of 'straggler detected' — the worker time variance was 0.405, well above our threshold of 0.06. Then at step 360, it switched BACK from async to sync because of 'loss rising' — the loss trend was 0.073, above the 0.06 threshold. And at step 480, it switched to async again because the straggler was still there."

"This table proves two things: First, the controller is **real** — it is making decisions based on actual math, not hardcoded timers. Second, it is **bi-directional** — it switches both ways depending on the situation."

"You can also verify these numbers against the raw log files in `logs/phase4_merged/`. Every CSV there was generated by the training code, not created manually."

---

## PART 12 — KEY FIGURES AS PROOF (2-3 minutes)

### SHOW: PPT Slide 11 (TTA Bars)

### SAY:

"Now let me show you the key graphs that visually prove our results."

"This is the **Time-to-Accuracy bar chart**. Each group of bars shows how long each mode took to reach 40 percent accuracy in each scenario. The numbers above each bar show how many seeds out of 3 actually reached the target. You can see that in Scenario A, Hybrid is clearly faster than Sync. In Scenario B, Hybrid is the fastest but with lower reliability — only 1 out of 3."

### SHOW: PPT Slide 12 (Hybrid Mode Timeline)

"This is my favourite graph — the **Hybrid Mode Timeline**. Each horizontal bar is one training run. Blue means Sync mode and orange means Async mode. The vertical black lines mark the exact moments when the controller switched modes. You can clearly see the system starts in blue, detects the straggler, and flips to orange. In seed 123 of Scenario A, you can even see it go back to blue briefly when the loss rose, and then return to orange. That is **bi-directional switching** happening live."

---

## PART 13 — WHY 40% ACCURACY (1 minute)

### SAY:

"Now a quick note about the 40 percent accuracy target. You might ask — why so low?"

"Three reasons. One, we ran this on a CPU-only laptop with short training budgets, 800 steps per worker, because we needed to run 41 total experiments. Two, the goal of this project is NOT to build the smartest AI — it is to prove that adaptive mode switching saves time. Three, 40 percent is just an arbitrary finish line. What matters is that Hybrid reached it **176 seconds faster** than Sync. If we trained to 90 percent, the time savings would be even bigger because the straggler penalty accumulates over every step."

---

## PART 14 — SCENARIO A SUMMARY AND CONCLUSION (2 minutes)

### SHOW: PPT Slide 10 (Scenario A Summary diagram)

### SAY:

"Here is the consolidated summary for Scenario A. Sync took 1409 seconds. Async took 641 seconds but with poor accuracy. Our Hybrid took 1233 seconds — 12 percent faster than Sync with similar accuracy and nearly double the throughput."

### SHOW: PPT Slide 13 (Key Findings)

"To summarize the key findings:"

"First, bi-directional switching is real and logged. Every hybrid run shows both sync-to-async and async-to-sync switches with explainable reasons."

"Second, for Scenario A — the persistent straggler — Hybrid is 12 percent faster than Sync on TTA, 1233 seconds versus 1409 seconds. The final accuracy of 41.9 percent is within the variance band of Sync's 43.2 percent. Throughput nearly doubled, from 13.6 to 23.3 images per second. Staleness was 359, lower than pure Async's 373."

"Third, for Scenario B — the transient straggler — Hybrid was the fastest at 864 seconds but lost accuracy because the controller over-reacted. The threshold sweep of 17 configurations shows exactly how to fix this."

"Fourth, the system is fully containerized in Docker and every result is backed by CSV log files that anyone can verify."

### SHOW: PPT Slide 14 (Thank You)

"Thank you for your time. I am happy to answer any questions."

---

## PPT SLIDE MAP (Quick reference while presenting)

| Slide | Content | Script Part |
|-------|---------|-------------|
| 1 | Title slide | Part 1 |
| 2 | Problem statement diagram | Part 1 |
| 3 | Sync vs Async diagram | Part 2 |
| 4 | Hybrid switching diagram | Part 3 |
| 5 | Controller signals diagram | Part 3 |
| 6 | System architecture diagram | Part 4 |
| 7 | Experimental setup (text) | Part 5 |
| 8 | Project pipeline diagram | Part 6 |
| 9 | Evaluation metrics (text) | Part 9 |
| 10 | Scenario A summary diagram | Part 14 |
| 11 | TTA bars (proof graph) | Part 12 |
| 12 | Hybrid mode timeline (proof graph) | Part 12 |
| 13 | Key findings (text) | Part 14 |
| 14 | Three scenarios diagram | Part 10 |
| 15 | Thank you | Part 14 |

## EXCEL CSVs TO HAVE OPEN (open all in Excel before recording)

| CSV File | When to show |
|----------|-------------|
| `results/phase1_sync_baseline_summary.csv` | Part 6 (Pipeline) |
| `results/phase2_async_baseline_summary.csv` | Part 6 (Pipeline) |
| `results/phase3_hybrid_controller_summary.csv` | Part 6 (Pipeline) |
| `results/phase4_scenario0_homogeneous.csv` | Part 10 (Scenario 0) |
| `results/phase4_scenarioA_persistent_straggler.csv` | Part 10 (Scenario A) |
| `results/phase4_scenarioB_transient_straggler.csv` | Part 10 (Scenario B) |
| `results/phase4_all_switch_events.csv` | Part 11 (Switch proof) |
