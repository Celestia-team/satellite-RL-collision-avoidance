# 🛰️ Celestia: RL for Secure Collision Avoidance in LEO

**Team:** Celestia-team  
**Project:** Reinforcement Learning for Secure Collision Avoidance Maneuvers  

## 👥 Team Members
- **Maryam Raouph** - Orbital Dynamics & TLE  
- **Aparna Dongale** - Gym Environment & RL
- **Maria Chowdhury** - Cybersecurity Layer
- **Pouya Latifiyan** - Training & Integration

## 👥 Team lead
- **Pouya Latifiyan** - Coordinator and Supervisor
- **Lucky Bosse** - Co-supervisor

## 🎯 Goal
Create a custom Gymnasium environment where an RL agent (PPO) learns 
Collision Avoidance Maneuvers (CAM) using real TLE data from poliastro, 
with cybersecurity constraints (spoofing/jamming simulation).


## 🚀 Training

```bash
# Python 3.10 required
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements-py310.txt
python train.py

## 🔧 Tools
- Python, poliastro, astropy, gymnasium
- stable-baselines3, matplotlib/plotly
- Celestrak TLE data

---
**Private Repository - Research Project**
