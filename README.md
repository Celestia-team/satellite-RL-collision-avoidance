# Domain Randomization for Adversarial Robustness in Orbital Collision Avoidance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PPO](https://img.shields.io/badge/RL-PPO-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Official implementation of **"Domain Randomization for Adversarial Robustness in Orbital Collision Avoidance: A Deep Reinforcement Learning Approach"** (ICGNC 2026).

## 🎯 Key Results

Our PPO-based framework achieves **93–100% collision avoidance success** across four orbital scenarios under severe cyber-attacks (σ = 20 km GPS spoofing), with emergent uncertainty-averse behavior.

### Performance Summary

| Scenario | Type | Clean Success | Adversarial Success | Safety Margin Change | Key Feature |
|----------|------|---------------|---------------------|---------------------|-------------|
| **SC01** | Classic Conjunction | 100% | 100% | **+16.9%** | Conservative expansion |
| **SC02** | High-Velocity | 100% | 100% | **+13.2%** | Velocity-responsive caution |
| **SC03** | Co-orbital | 100% | 100% | **-10.0%** | Tight precision control |
| **SC04** | Long-Term Drift | 93.3% | **100%** | **+19.1%** | Recovery under attack |

*Adversarial conditions: Gaussian noise (σ = 20 km) + fixed spoofing bias [20, -15, 10] km*

### Robustness Metrics

- **Collision Rate**: 0% under all attack severities (Clean to Severe)
- **Fuel Consumption**: ~86 m/s average (< 3% variation across attacks)
- **Safety Margin**: Emergent expansion up to +19.1% without explicit risk-sensitive training
- **Real-world Validation**: 97.5% success on 108 Iridium 33 debris objects (SGP4 propagation)

### Attack Resilience

| Attack Level | Noise σ (km) | Spoofing Probability | Success Rate | Avg. Reward |
|--------------|--------------|-------------------|----------------|-------------|
| Clean | 0 | 0% | 98.3% | 416.7 |
| Moderate | 5 | 30% | 100% | 412.9 |
| Aggressive | 10 | 50% | 100% | 398.4 |
| Severe | 20 | 70% | 100% | 394.1 |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Celestia-team/satellite-RL-collision-avoidance.git
cd satellite-RL-collision-avoidance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Training

Train on specific scenario with domain randomization:
```bash
python train.py --scenario SC01_CLASSIC --steps 400000 --attack-randomization
```

Train all 4 scenarios (100k steps each):
```bash
python train.py --scenario all --steps 400000
```

### Evaluation

Evaluate under adversarial conditions:
```bash
python evaluate_real_orbital.py --model models/ppo_cam.zip --scenario SC01_CLASSIC --attack-level severe
```

Validate on real TLE data (Iridium 33):
```bash
python validate_real_tle.py --model models/ppo_cam.zip --tle-data data/iridium-33-debris.txt
```

### Generate Paper Figures

```bash
python generate_figures.py --results results/security_results.json --output figures/
```

This generates:
- `fig1_success_rate.png`: Success rates across attack levels
- `fig2_min_distance.png`: Safety margin modulation
- `fig3_tle_validation.png`: Real debris validation

## 📁 Repository Structure

```
satellite-RL-collision-avoidance/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Excludes venv, __pycache__, large models
│
├── train.py                 # Main training script (PPO + Domain Randomization)
├── evaluate_real_orbital.py  # Evaluation with SGP4 dynamics
├── validate_real_tle.py     # Validation on Iridium 33 catalog
├── generate_figures.py      # Plot generation for paper
├── generate_scenarios.py    # JSON scenario generator
│
├── src/                     # Core modules
│   ├── environment.py       # SatelliteCAMEnv (Gym interface)
│   ├── orbital_dynamics.py  # SGP4 + CW dynamics
│   ├── security_module.py   # Adversarial attack wrapper
│   └── evaluation_script.py # Metrics computation
│
├── scenarios/               # 20 pre-configured scenarios (4 scenarios × 5 attack levels)
│   ├── index.json          # Scenario metadata
│   ├── SC01_CLASSIC_clean.json
│   ├── SC01_CLASSIC_severe_attack.json
│   ├── SC02_HIGHVEL_clean.json
│   └── ...
│
├── results/                 # Evaluation outputs (JSON)
├── figures/                 # Generated plots (PNG/PDF)
├── models/                  # Trained policies (.zip)
├── data/                    # TLE data and debris catalogs
└── docs/                    # Additional documentation
```

## 🛰️ Scenarios Description

### SC01: Classic Conjunction
- **Geometry**: 50 km initial separation, 4-hour TCA
- **Based on**: 2009 Iridium-Cosmos collision
- **Velocity**: ~10 km/s relative
- **Challenge**: Standard CAM scenario

### SC02: High-Velocity
- **Geometry**: 86.4° vs 85.99° inclination difference
- **Velocity**: ~14 km/s relative (high-speed approach)
- **Challenge**: Time-critical decision making

### SC03: Co-orbital
- **Geometry**: e < 0.001 (circular orbits)
- **Velocity**: <100 m/s relative (slow approach)
- **Challenge**: Precision low-thrust control

### SC04: Long-Term Drift
- **Geometry**: 72-hour propagation
- **Initial**: Dispersed conditions
- **Challenge**: Long-horizon planning under uncertainty

## 📊 Detailed Results

### Emergent Uncertainty-Averse Behavior

Under adversarial conditions, the policy automatically expands safety margins without explicit risk-sensitive reward shaping:

```
SC01 (Classic):     Normal margin → +16.9% expansion
SC02 (High-Vel):    Normal margin → +13.2% expansion  
SC03 (Co-orbital):  Normal margin → -10.0% (tighter but controlled)
SC04 (Drift):       Normal margin → +19.1% expansion (strongest)
```

### Fuel Efficiency

| Scenario | Clean ΔV (m/s) | Adversarial ΔV (m/s) | Variation |
|----------|----------------|---------------------|-----------|
| SC01 | 85.4 | 86.2 | +0.9% |
| SC02 | 87.1 | 87.9 | +0.9% |
| SC03 | 84.8 | 85.1 | +0.4% |
| SC04 | 86.3 | 86.0 | -0.3% |

*Fuel consumption remains stable despite severe attacks, demonstrating robustness without excessive fuel expenditure.*

### Real TLE Validation (Iridium 33)

Validation on 108 cataloged debris fragments:

| Metric | Training (CW) | Validation (SGP4) | Difference |
|--------|---------------|-------------------|------------|
| Success Rate | 98.3% | 97.5% | -0.8% |
| Avg. Miss Distance | 2.31 km | 2.28 km | -1.3% |
| Avg. Fuel | 86.2 m/s | 87.9 m/s | +2.0% |

*Successful sim-to-real transfer confirmed.*

## 🔬 Methodology Highlights

- **Domain Randomization**: Training with randomized attack parameters (σ ∈ {0,5,10,20} km, p_attack ∈ {0.0,0.3,0.5,0.7})
- **Observation Space**: 7D state (relative position/velocity in LVLH + distance)
- **Action Space**: Continuous ΔV in RIC coordinates (bounded ±0.01 km/s)
- **Reward**: Multi-objective (collision penalty -1000, fuel cost -0.1, success bonus +100)
- **Architecture**: Separate Actor-Critic MLPs (256×256, ReLU)

## 📄 Citation

```bibtex
@inproceedings{latifiyan2026domain,
  title={Domain Randomization for Adversarial Robustness in Orbital Collision Avoidance: A Deep Reinforcement Learning Approach},
  author={Latifiyan, Pouya and Raouph, Maryam and Dongale, Aparna and Chowdhury, Maria and Bosse, Lucky},
  booktitle={Proceedings of the International Conference on Guidance, Navigation and Control (ICGNC)},
  year={2026},
  organization={Springer}
}
```

## 📧 Contact

- **Pouya Latifiyan**: Pouya@buaa.edu.cn (Corresponding Author)
- **Maryam Raouph**: Maryam.Raouph@staff.uni-marburg.de
- **Aparna Dongale**: MIT Academy of Engineering, Pune
- **Maria Chowdhury**: University of the Potomac, Washington, D.C.
- **Lucky Bosse**: Beihang University, Beijing (boselucky@buaa.edu.cn)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Iridium 33 TLE data from [Celestrak](https://celestrak.org/)
- SGP4 propagation using `sgp4` Python library
- RL framework based on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
```
