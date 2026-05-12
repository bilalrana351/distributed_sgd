# Project Overview: Hybrid Distributed SGD

This document explains what we built for the Parallel and Distributed Computing project, how it was implemented, the architecture, the pipeline, the metrics used, and the final results obtained. It uses very simple English so it is easy to present and understand.

## 1. What I Did (The Implementation)
In deep learning, training huge models requires multiple workers working at the same time. These workers need to share their updates (gradients) with each other. There are two standard ways to do this:
1. **Synchronous (Sync) Mode:** All workers calculate their updates, wait for everyone to finish, average the updates, and then take one step. 
   - *Equation:* $W_{t+1} = W_t - \eta \frac{1}{N} \sum_{i=1}^N \nabla L_i(W_t)$
   - *Problem:* If one worker is slow (a straggler), everyone else has to wait.
2. **Asynchronous (Async) Mode:** Every worker calculates its update and sends it to a central server immediately. The server updates the model without waiting for anyone.
   - *Equation:* $W_{t+1} = W_t - \eta \nabla L_i(W_{t-\tau})$
   - *Problem:* Because nobody waits, some workers might use old, "stale" weights ($\tau$ is the staleness). This can ruin the model's accuracy.

**My Implementation:** I created a **Hybrid System** that sits exactly in the middle. It monitors the workers live. If everything is running smoothly, it uses the Sync mode to ensure high accuracy. But if a worker suddenly becomes slow, my system detects it and instantly switches to Async mode so the fast workers don't have to wait. If the model starts losing accuracy because of Async mode, it switches back to Sync mode to stabilize.

## 2. The Pipeline and Architecture
The pipeline of this project is executed on a single machine using Docker to simulate a distributed environment.
1. **Workers** read the data (CIFAR-10) and compute gradients.
2. **Gradient Queue:** Workers push their gradients to a shared queue.
3. **Parameter Server (PS):** A central process that holds the main model.
4. **Aggregator:** Depending on the mode, it either waits for all workers to average their gradients (Sync) or applies them immediately (Async).
5. **Controller:** Monitors the signals and switches the mode when necessary.

### Architecture Diagram Explained
The system uses a 1-container layout. Inside this container, we have:
- **Parameter Server (PS):** This is the boss. It has three main parts:
  - *Aggregator:* Takes the updates from the workers and applies them to the main model.
  - *Metrics Collector:* Keeps an eye on how much time workers are taking, how big the gradients are, and if the loss is going up or down.
  - *Controller:* The brain. It looks at the metrics and decides, "Should we be in Sync or Async mode right now?"
- **Workers (1 to N):** These are the processes doing the hard work. They take a piece of the dataset, calculate the math (gradients), and send it to the PS via a Queue.
- **Shared Memory:** Both the PS and the Workers can see the model weights in the shared memory so they don't have to copy huge files around.

## 3. Emphasis on Hybrid Switching
The magic of this project is the **Bi-directional Online Controller**. 
- It switches from **Sync to Async** when it detects a straggler (a slow worker). The signal used is `worker_time_var` (the variance in how long workers take).
- It switches from **Async to Sync** when it detects the training is getting unstable. The signals used are `grad_norm_var` (gradients becoming too chaotic) and `loss_trend` (the loss going up instead of down).

## 4. Metrics Used
To measure if our system is actually good, we tracked several metrics:
- **Time-to-Accuracy (TTA):** How many seconds it takes to reach our target of 40% accuracy.
- **Wall Clock Time:** Total time the experiment ran.
- **Throughput:** How many images the system processed per second.
- **Staleness:** How outdated a worker's model was when it sent an update.
- **Switches:** How many times the controller flipped between Sync and Async.

## 5. Final Results and Outputs
We tested the system under different scenarios (Normal, Persistent Slow Worker, Transient Spikes). Here is what we found:
- **When there is a slow worker (Scenario A):** Pure Sync mode took **1409 seconds** to reach 40% accuracy. Our Hybrid mode took only **1233 seconds** and achieved roughly the same final accuracy. This proves the system works exactly as intended! It saved time without losing quality.
- **Throughput:** Hybrid mode processed ~23 images per second compared to Sync's ~13 images per second.
- **Code Output:** During runtime, the code generates CSV logs detailing every single switch, showing *why* it switched (e.g., "straggler_detected" or "loss_rising"). These prove that the system is dynamically reacting to the environment.
