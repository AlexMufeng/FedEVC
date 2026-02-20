

# FedEVC: Federated Electric Vehicle Charging Demand Forecasting under Hybrid Operations

Official PyTorch implementation of "FedEVC: Federated Electric Vehicle Charging Demand Forecasting under Hybrid Operations"

## Datasets

We test on three real-world EV charging demand datasets collected from three cities in China: Beijing, Hangzhou, and Dongguan. For each city, we randomly sample 100 charging stations, covering both exclusive and roaming types, to serve as representative nodes. The data span July 1, 2025 to November 30, 2025 (153 days). Raw charging records are logged by the electric vehicle supply equipment at each station and aggregated into 30-minute intervals. We construct two station-level spatial-temporal sequences:

(1) Orders: The number of charging sessions initiated per interval (relevant to station recommendation and dynamic pricing); 

(2) Energy: The total delivered energy per interval (relevant to grid load analysis and scheduling). 

## Environment

All methods are implemented in PyTorch 1.13.1 and run on a server with an Intel Xeon Gold 6230R CPU (2.10 GHz) and four NVIDIA A100 GPUs.

```bash
conda create -n fedevc "python=3.11"
conda activate fedevc
bash install.sh
```

## Run

```bash
bash run.sh
```

