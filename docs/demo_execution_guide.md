# Demo Execution Guide: Step-by-Step

This document is your step-by-step cheat sheet for presenting the project during your demo. Keep this open on your screen or printed out beside you.

## Preparation Before the Demo
1. Make sure Docker is installed and running on your Windows machine.
2. Open your terminal (PowerShell or Command Prompt) and navigate to the project directory:
   ```powershell
   cd c:\Users\786\Documents\pdc_project
   ```
3. *(Optional but recommended)* Build the Docker container just to be safe so it doesn't download things during the presentation:
   ```powershell
   docker compose -f docker/docker-compose.yml build
   ```

---

## Step 1: Introduce the Project (2-3 Minutes)
- **What to say:** "Hi everyone, for my PDC project, I built a Hybrid Distributed SGD system. Normally, training uses either Sync mode (which is accurate but gets stuck if one worker is slow) or Async mode (which is fast but loses accuracy because of stale data). I built a system that dynamically switches between both during runtime based on live metrics."
- **Show the Architecture:** Open the architecture diagram (from `docs/project_overview_demo.md` or your slides). 
- **Explain the Diagram:** "We have a Parameter Server (the boss) and Workers. The workers calculate gradients and send them. The magic happens in the Controller inside the server. It watches the workers. If a worker gets slow, it switches to Async. If the loss goes bad, it switches back to Sync."

## Step 2: Show the Code Pipeline Briefly (1-2 Minutes)
- **What to show:** Open `src/hybrid/controller.py` and `src/hybrid/train_hybrid.py` in your code editor.
- **What to say:** "Here is the core logic. As you can see, the controller continuously calculates the variance in worker times and gradient norms. Based on thresholds, it flips a `mode_flag`. When the flag is 0, the aggregator waits for everyone. When it is 1, it applies updates immediately."

## Step 3: Run the Live Demo (3-5 Minutes)
Instead of waiting 20 minutes for a full training run, we will run the pre-configured Hybrid scenario with a persistent straggler (Scenario A).

- **Execute this command in your terminal:**
  ```powershell
  docker compose -f docker/docker-compose.yml run --rm scenario_runner src/hybrid/launcher.py --nworkers 2 --config configs/phase4_sA_hybrid.yaml
  ```
- **What to say while it runs:** "I am now spinning up a Docker container that simulates a distributed network. It creates one Parameter Server and two workers. I have purposely made one worker very slow to act as a straggler."
- **Point to the terminal output:** As the code runs, point out to the examiner when the terminal prints that it is switching modes. "Look here! The controller just detected the slow worker and switched to Async mode so the fast worker isn't blocked."

## Step 4: Show the Results and Logs (2-3 Minutes)
Once the run finishes, or by showing the already generated files:

- **Open the Logs:** Go to the `logs/` folder and open the `_switches.csv` file for the run you just did.
- **What to say:** "This is the proof that the dynamic switching works. In this CSV, you can see the exact steps where the controller flipped the mode. It says `straggler_detected` when it switched to Async, and `loss_rising` when it needed to stabilize and switched back to Sync."

## Step 5: Show the Final Graphs and Metrics (2 Minutes)
- **What to show:** Open the images in `docs/figures/`, specifically `phase5_tta_bars.png` and `phase5_hybrid_mode_timeline.png`.
- **What to say:** 
  1. *Show Timeline:* "This timeline graph is beautiful. The black lines show exactly when the system switched modes during training. You can see it spends most of its time in Async to be fast, but occasionally dips into Sync to stay accurate."
  2. *Show TTA Graph:* "Here is the final result. In the scenario with a slow worker, standard Sync mode took 1409 seconds to reach 40% accuracy. Our Hybrid system took only 1233 seconds, proving we saved significant time while maintaining the same accuracy level!"

## Step 6: Conclusion
- **What to say:** "In conclusion, the pipeline was successfully containerized, the bi-directional switching works live based on real math (variance and trends), and the final metrics prove it solves the straggler problem efficiently. Thank you, I am happy to answer any questions."
