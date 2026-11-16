# 🚀 CPU Cluster Lab — Complete End-to-End Field Walkthrough
**Document Version:** 1.0.0  
**Target Environment:** Bare-Metal Commodity PCs, Ubuntu Server 22.04 LTS, Gigabit Ethernet LAN  
**Master Node (Node 01):** Coordinator & Primary Rank (`rank=0`)  
**Worker Node (Node 02):** Compute Worker (`rank=1`)  
**Control Station:** Personal Windows Laptop (via SSH)

---

## Table of Contents
1. [Phase 0 — Pre-Lab Preparation (Tonight)](#phase-0--pre-lab-preparation-tonight)
2. [Phase 1 — Bare-Metal OS Installation (Ubuntu Server 22.04 LTS)](#phase-1--bare-metal-os-installation-ubuntu-server-2204-lts)
3. [Phase 2 — First Boot & Network Verification](#phase-2--first-boot--network-verification)
4. [Phase 3 — Control Station Setup (SSH from Windows Laptop)](#phase-3--control-station-setup-ssh-from-windows-laptop)
5. [Phase 4 — Repository Deployment & Environment Setup](#phase-4--repository-deployment--environment-setup)
6. [Phase 5 — Milestone 1: Distributed Process Group Verification (Smoke Test)](#phase-5--milestone-1-distributed-process-group-verification-smoke-test)
7. [Phase 6 — Milestone 2: Distributed Training Demonstrations (CIFAR-10 & GPT-2)](#phase-6--milestone-2-distributed-training-demonstrations-cifar-10--gpt-2)
8. [Phase 7 — Milestone 3: Benchmarking & Generating Publication Figures](#phase-7--milestone-3-benchmarking--generating-publication-figures)
9. [Phase 8 — Executive Presentation Script for Your Professor](#phase-8--executive-presentation-script-for-your-professor)
10. [Troubleshooting & Emergency Contingency Procedures](#troubleshooting--emergency-contingency-procedures)

---

## Phase 0 — Pre-Lab Preparation (Tonight)

### Hardware Checklist
- [ ] 1× USB Flash Drive (8 GB or larger, formatted FAT32/exFAT).
- [ ] 2× Cat5e or Cat6 RJ-45 Ethernet Patch Cables.
- [ ] Personal Windows Laptop + Charger.
- [ ] Physical access permissions to 2 lab PCs connected to the lab router/switch.

### Bootable USB Creation
1. Download **Ubuntu Server 22.04.4 LTS (64-bit x86_64)** ISO from:  
   👉 `https://releases.ubuntu.com/22.04/ubuntu-22.04.4-live-server-amd64.iso`
2. Download **Rufus** (portable executable) from:  
   👉 `https://rufus.ie`
3. Insert USB drive, launch Rufus, and set:
   - **Device:** Your USB drive
   - **Boot selection:** Select the downloaded Ubuntu `.iso`
   - **Partition scheme:** `GPT` (for modern UEFI) or `MBR` (if older legacy BIOS)
   - **Target system:** `UEFI (non-CSM)`
4. Click **START** -> Select **Write in ISO Image mode (Recommended)** -> OK. Wait for completion.

---

## Phase 1 — Bare-Metal OS Installation (Ubuntu Server 22.04 LTS)

Repeat this exact workflow sequentially on **Node 01**, then **Node 02**.

### 1. BIOS/UEFI Boot
1. Plug the Ethernet cable from the lab switch/router into the PC's Ethernet port.
2. Insert the bootable USB drive into a rear USB 3.0 port.
3. Power on the PC and immediately tap the **Boot Menu Key** (typically `F12`, `F11`, `F9`, or `F8` depending on PC brand: Dell=`F12`, HP=`F9`, Lenovo=`F12`, Acer=`F12`).
4. Select the UEFI USB Flash Drive and hit **Enter**.
5. At the GRUB menu, select **Try or Install Ubuntu Server** and press **Enter**.

### 2. Subiquity Text Installer Walkthrough
*Navigate with arrow keys, select with `Spacebar`, confirm with `Enter`.*

1. **Language:** Select `English` -> Enter.
2. **Keyboard Configuration:** Layout: `English (US)` -> Done.
3. **Choose type of install:** Select `Ubuntu Server` (do **not** choose minimized).
4. **Network connections:** 
   - Verify the `eth0` / `enp3s0` interface shows an assigned IPv4 address via DHCP (e.g., `192.168.1.xxx`).
   - If connected, select **Done**.
5. **Configure proxy:** Leave blank unless your university requires an HTTP proxy -> Done.
6. **Configure Ubuntu archive mirror:** Keep default -> Done.
7. **Guided storage configuration:**
   - Select **Use an entire disk**.
   - *(Uncheck "Set up this disk as an LVM group" unless needed; standard ext4 simplifies recovery).*
   - Select your target SSD/HDD -> Done -> Confirm destruction of data.
8. **Profile setup (CRITICAL — Use identical user credentials on both nodes):**
   - **Your name:** `cluster`
   - **Your server's name:** `node01` *(for PC 1)* | `node02` *(for PC 2)*
   - **Pick a username:** `cluster`
   - **Choose a password:** *(set a secure, memorable password, e.g., `cluster123`)*
   - **Confirm password:** *(re-type password)* -> Done.
9. **Upgrade to Ubuntu Pro:** Skip for now -> Continue.
10. **SSH Setup (CRITICAL):**
    - Tap `Spacebar` to enable: `[*] Install OpenSSH server`.
    - Import SSH identity: `No` -> Done.
11. **Featured Server Snaps:** Leave all unchecked -> Done.
12. **System installation:** Wait ~5–10 minutes for packages to unpack and install.
13. When complete, select **Reboot Now**. Unplug the USB drive when prompted and hit **Enter**.

---

## Phase 2 — First Boot & Network Verification

Log in at the physical console of each machine with username `cluster` and your password.

### 1. Identify Network Interface & IP Address
Run on both machines:
```bash
ip -br addr show
```
*Look for your Ethernet interface (`eth0`, `enp2s0`, etc.). Example output:*
```text
enp3s0   UP   192.168.1.101/24
```
Write down these IP addresses:
- **Node 01 (Master):** `192.168.1.101` *(example)*
- **Node 02 (Worker):** `192.168.1.102` *(example)*

### 2. Bidirectional Ping Verification
From **Node 01**, ping Node 02:
```bash
ping -c 3 192.168.1.102
```
From **Node 02**, ping Node 01:
```bash
ping -c 3 192.168.1.101
```
*Requirement: 0% packet loss and latency under < 0.5 ms.*

---

## Phase 3 — Control Station Setup (SSH from Windows Laptop)

Once logged into both PCs, connect your Windows Laptop to the same router (via Ethernet or lab Wi-Fi). You can now step away from the physical lab PC monitors and manage everything from your laptop.

### 1. Open Two PowerShell Windows Side-by-Side
Open two PowerShell or Windows Terminal windows on your Windows laptop:

**Terminal Window 1 (Master Controller):**
```powershell
ssh cluster@192.168.1.101
```
*(Type `yes` when prompted about host authenticity, then enter your password).*

**Terminal Window 2 (Worker Controller):**
```powershell
ssh cluster@192.168.1.102
```

### 2. Configure Passwordless SSH Between Nodes (Node 01 -> Node 02)
In **Terminal Window 1 (Node 01)**, execute:
```bash
# Generate high-security RSA keypair (Press Enter for all prompts)
ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa

# Authorize Node 01 to connect to Node 02 without a password
ssh-copy-id cluster@192.168.1.102
```
Verify passwordless login:
```bash
ssh cluster@192.168.1.102 "hostname"
# Should instantly output: node02
```

---

## Phase 4 — Repository Deployment & Environment Setup

### 1. Clone Repository onto Node 01
In **Terminal Window 1 (Node 01)**:
```bash
cd ~
git clone https://github.com/YOUR_USERNAME/CPU-Cluster.git cpu-cluster
cd cpu-cluster
```

### 2. Execute Automated Environment Provisioning
Run the setup script on **Node 01**:
```bash
bash setup.sh
```
*This installs Python 3.11, venv, CPU-optimized PyTorch 2.3, TensorBoard, and creates project folders.*

### 3. Propagate Codebase & Run Setup on Node 02
From **Terminal Window 1 (Node 01)**, push the codebase to Node 02 and trigger its installer via SSH:
```bash
# Push project directory to Node 02
scp -r ~/cpu-cluster cluster@192.168.1.102:~/

# Trigger remote provisioning on Node 02
ssh cluster@192.168.1.102 "cd ~/cpu-cluster && bash setup.sh"
```

### 4. Set Master IP Address in `config.py`
Open `config.py` in your editor on **both nodes**:
```bash
nano config.py
```
Update line 11 with Node 01's exact IP:
```python
MASTER_ADDR = "192.168.1.101"   # Use Node 01's actual IP
```
Save and exit (`Ctrl+O` -> `Enter` -> `Ctrl+X`).

Sync the updated config to Node 02:
```bash
scp config.py cluster@192.168.1.102:~/cpu-cluster/config.py
```

---

## Phase 5 — Milestone 1: Distributed Process Group Verification (Smoke Test)

This step proves inter-node collective communications (`AllReduce`, `Broadcast`, `Barrier`) operate correctly using PyTorch's `c10d` rendezvous service over Gloo.

### Execution
1. In **Terminal Window 1 (Node 01)**:
   ```bash
   bash launch_master.sh 01_smoke_test.py
   ```
   *Terminal displays:* `Waiting for 1 worker(s)...`

2. Within 30 seconds, in **Terminal Window 2 (Node 02)**:
   ```bash
   bash launch_worker.sh 01_smoke_test.py 1
   ```

### Expected Output
Both terminals synchronize and print:
```text
============================================================
  CLUSTER SMOKE TEST — CPU Cluster Lab
  Nodes   : 2
  Backend : gloo
  Master  : 192.168.1.101:29500
============================================================
  ✓  Node 0 (node01) — connected
  ✓  Node 1 (node02) — connected

  [Test 1] AllReduce SUM         : PASS ✓
           Expected=3  Got=3
  [Test 2] Broadcast (rank 0→all): PASS ✓
  [Test 3] Barrier sync          : PASS ✓
  [Test 4] Gradient sync sim     : PASS ✓

============================================================
  RESULT : ALL TESTS PASSED — CLUSTER IS WORKING ✓
           2-node CPU cluster verified and ready
============================================================
```
🎉 **Milestone 1 Completed! Take a screenshot.**

---

## Phase 6 — Milestone 2: Distributed Training Demonstrations (CIFAR-10 & GPT-2)

### 1. Vision Benchmark: ResNet-18 on CIFAR-10
*Demonstrates standard data-parallel CV training across commodity nodes.*

In **Terminal Window 1 (Node 01)**:
```bash
bash launch_master.sh 02_train_cifar.py
```
In **Terminal Window 2 (Node 02)**:
```bash
bash launch_worker.sh 02_train_cifar.py 1
```

**Observation:**
- Node 01 and Node 02 automatically download and partition the CIFAR-10 dataset.
- Epoch logs print every ~2 minutes with loss, accuracy, and images/sec throughput.
- Checkpoints and CSV logs are persisted to `models/` and `results/`.

---

### 2. Language Benchmark: GPT-2 Small on WikiText-2
*Direct replication of the base paper's objective on commodity x86 hardware.*

In **Terminal Window 1 (Node 01)**:
```bash
bash launch_master.sh 03_train_gpt2.py
```
In **Terminal Window 2 (Node 02)**:
```bash
bash launch_worker.sh 03_train_gpt2.py 1
```

**Observation:**
- Initializes 117M parameter Transformer architecture in CPU RAM.
- Tokens/sec throughput is recorded after each gradient AllReduce step.
- Output logged to `results/gpt2_2nodes_<timestamp>.csv`.

---

### 3. Optional: Live TensorBoard Visuals on Windows Laptop
In a new PowerShell tab on your Windows Laptop, open an SSH port-forward tunnel:
```powershell
ssh -L 6006:localhost:6006 cluster@192.168.1.101
```
Inside this session on Node 01, launch TensorBoard:
```bash
source ~/cpu-cluster/.venv/bin/activate
tensorboard --logdir=~/cpu-cluster/logs/ --port=6006
```
Open your laptop browser to:  
👉 **`http://localhost:6006`**  
*You will see real-time loss curves, learning rate progression, and token throughput live charts.*

---

## Phase 7 — Milestone 3: Benchmarking & Generating Publication Figures

### 1. Standardized Multi-Run Benchmark
To capture stable, publication-grade numbers:

In **Terminal Window 1 (Node 01)**:
```bash
bash launch_master.sh 04_benchmark.py
```
In **Terminal Window 2 (Node 02)**:
```bash
bash launch_worker.sh 04_benchmark.py 1
```
*Generates `results/benchmark_2nodes_<timestamp>.csv` and an executive text summary.*

### 2. Single-Node Baseline (Crucial for Scaling Efficiency Calculation)
Run on **Node 01 alone** (no worker required) to record the baseline:
```bash
bash launch_single.sh 04_benchmark.py
```

### 3. Automated Figure Generation
On Node 01, run the plotting pipeline:
```bash
source .venv/bin/activate
python3 src/05_plot_results.py
```
*Inspect generated PNG figures in `results/figures/`:*
- `01_loss_curves.png`
- `02_throughput_bar.png`
- `03_scaling_efficiency.png`
- `04_network_bandwidth.png`

---

## Phase 8 — Executive Presentation Script for Your Professor

When demonstrating the cluster to your professor, follow this narrative:

1. **Context & Motivation:**
   > *"Professor, the 2025 base paper demonstrated distributed LLM training on Apple Silicon using proprietary unified memory and high-cost Thunderbolt 5 interconnects. Our research tests whether commodity x86 CPUs and standard Gigabit Ethernet can achieve viable training throughput using pure open-source primitives."*

2. **Infrastructure Walkthrough:**
   > *"We have deployed Ubuntu Server 22.04 on these bare-metal machines, networked through standard Cat6 cables. The master node is coordinating via PyTorch's `c10d` rendezvous service, and communication is executed through the CPU-native Gloo backend without requiring proprietary hardware."*

3. **Live Demonstration (Launch 01_smoke_test.py):**
   > *"Here in Terminal 1 is Node 01 (Rank 0) and in Terminal 2 is Node 02 (Rank 1). When we initiate the smoke test, both machines execute an AllReduce SUM across the network interface. As you can see, both nodes return `Result: 3`, confirming zero packet drop, functional synchronization, and barrier locks."*

4. **Training Demonstration (Launch 02_train_cifar.py or 03_train_gpt2.py):**
   > *"During live training, each node processes independent mini-batches. After backward pass computation, their gradient tensors are reduced and averaged across Ethernet before parameter updates. The system outputs real-time CSV metrics and TensorBoard telemetries."*

5. **Next Steps (Scaling to 10–15 Lab PCs):**
   > *"Because our launcher accepts dynamic `--nnodes` and `--node-rank` parameters, this exact architecture scales directly to 10, 15, or 30 lab PCs simply by updating the node count. Our next milestone is mapping the scaling efficiency curve as we add more nodes."*

---

## Troubleshooting & Emergency Contingency Procedures

| Symptom | Root Cause | Immediate Remediation |
| :--- | :--- | :--- |
| `Connection refused: port 29500` | Ubuntu Firewall blocking rendezvous port | Run on both nodes:<br>`sudo ufw allow 29500/tcp` |
| `RuntimeError: Gloo connectFullMesh timed out` | Incorrect `MASTER_ADDR` or subnet isolation | Check `config.py` on both nodes; run `ping <node01-ip>` from Node 02. |
| `Host is unreachable` during SSH | Lab router isolated client devices or Wi-Fi AP isolation active | Ensure both PCs are plugged into physical switch LAN ports; avoid Wi-Fi. |
| `torchrun: command not found` | Virtual environment not active | Execute: `source ~/cpu-cluster/.venv/bin/activate` |
| `MemoryError / Killed` during training | Batch size exceeds available physical RAM | Edit `config.py`: reduce `DEFAULT_BATCH_SIZE = 16` or `GPT2_BATCH_SIZE = 2`. |
| Outdated code on Node 02 | Changes made to Node 01 were not copied | Run on Node 01:<br>`scp -r ~/cpu-cluster/src ~/cpu-cluster/config.py cluster@192.168.1.102:~/cpu-cluster/` |

---
*CPU Cluster Lab Research Walkthrough — Prepared for University Field Deployment*
