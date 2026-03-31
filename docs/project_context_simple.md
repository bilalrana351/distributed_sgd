# Project Context (Simple and Concise)

## What we are building

We are building a **hybrid distributed SGD training system**.

In simple words:
- We train one model using multiple workers (nodes/containers).
- The system can run in two modes:
  1. **Synchronous Ring All-Reduce (Sync mode)**
  2. **Asynchronous Parameter Server (Async mode)**
- Our system will **switch between these two modes at runtime** based on what is happening during training.

---

## Basic idea: SGD and distributed training

- **SGD (Stochastic Gradient Descent)** updates model weights step by step using small data batches.
- In **distributed SGD**, many workers train in parallel and must share updates.
- This makes training faster, but communication between workers becomes a big challenge.

Main issue:
- If communication is slow, training speed drops a lot.
- If updates are not well synchronized, model quality can drop.

---

## What existing frameworks do

### 1) Parameter Server (Async)
- Workers send gradients to a central server.
- Server updates global model and workers pull new parameters.
- Good: does not wait for slow workers (better hardware use).
- Bad: updates can be stale, which can hurt final accuracy.

### 2) Ring All-Reduce (Sync)
- Workers directly exchange and reduce gradients in a ring.
- Good: strong consistency and usually better training stability/accuracy.
- Bad: all workers must wait for the slowest worker (straggler problem).

---

## Gap we found

Current systems mostly force a fixed choice:
- either sync (stable but can be slow with stragglers),
- or async (faster but can lose accuracy due to staleness).

Even hybrid ideas in literature are often limited (for example, one-way switching or assumptions of fixed/persistent stragglers).

Real systems have **changing conditions**:
- sometimes network latency spikes,
- sometimes workers become slow only for short periods (transient heterogeneity),
- sometimes training becomes unstable and needs stricter sync.

So the missing thing is:
- **real-time, bi-directional adaptation** between sync and async modes.

---

## Our improvement

We propose a controller that monitors:
- communication latency,
- gradient variance/diversity (training stability signal),
- (optional) staleness signal.

Then it decides:
- switch to **async** when waiting cost is high (to avoid slowdown),
- switch to **sync** when training stability/consistency needs to be stronger.

This is our core system-level contribution.

---

## Goal

Our goal is to reduce **Time-to-Convergence (TTC)** while keeping accuracy close to sync baseline.

Success means:
- faster convergence than pure sync under stragglers/transient delays,
- similar final accuracy to sync baseline,
- better resource use (less idle waiting),
- stable switching behavior (not switching too frequently).

---

## Experiment setup (practical scope)

- Single laptop/workstation.
- 4 to 8 Docker containers as emulated distributed nodes.
- PyTorch DDP for sync mode.
- PyTorch RPC for async parameter server mode.
- CIFAR-10/100 with ResNet-18 or MobileNetV2.
- Test in three scenarios:
  1. Homogeneous baseline
  2. Persistent stragglers
  3. Transient random latency spikes

---

## One-line summary

This project builds an adaptive distributed SGD system that switches between Ring All-Reduce and Parameter Server modes in real time to get both speed and stability under changing network/node conditions.

