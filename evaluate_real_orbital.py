#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stable_baselines3 import PPO
from environment import SatelliteCAMEnv

# Load trained model (simplified) but test on real orbital
model = PPO.load("./models/ppo_satellite_final")

# Test on real orbital with fallback handling
env_real = SatelliteCAMEnv(
    scenario_file="scenarios/SC01_CLASSIC_orbit.json",
    use_real_orbital=True,
    max_steps=100,
    max_deviation=500.0  # Allow large deviation for far debris
)

obs, info = env_real.reset()
print(f"Real orbital test: Initial distance = {info.get('distance_km', 'N/A')} km")

# Run manual test
for step in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env_real.step(action)
    print(f"Step {step+1}: dist={info.get('distance_km', 0):.1f}km, reward={reward:.1f}")
    if done:
        break