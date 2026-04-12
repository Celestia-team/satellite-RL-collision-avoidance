"""
Satellite Collision Avoidance Environment (CAM)
Full integration with orbital_dynamics from Dr. Raouph
Author: Aparna (Person 1) - Refactored by Celestia Team
"""

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path

# Import from Dr. Raouph's module
try:
    from orbital_dynamics import (
        load_tle_from_celestrak,
        create_orbit_from_tle,
        propagate_orbit,
        calculate_relative_state,
        apply_impulsive_maneuver,
        detect_close_approach
    )
    POLIASTRO_AVAILABLE = True
except ImportError:
    POLIASTRO_AVAILABLE = False
    print("Warning: orbital_dynamics not available, using simplified model")


def parse_tle_file(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Parse a TLE file containing two satellite objects.
    Returns: (chaser_tle, threat_tle) where each is [name, line1, line2]
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    tles = []
    i = 0
    
    while i < len(lines):
        if lines[i].startswith('1 ') and i >= 2:
            name = lines[i-2]
            line1 = lines[i]
            line2 = lines[i+1] if i+1 < len(lines) else ""
            
            if line2.startswith('2 '):
                tles.append([name, line1, line2])
                i += 2
        i += 1
    
    if len(tles) < 2:
        raise ValueError(f"Expected 2 TLE objects, found {len(tles)} in {filepath}")
    
    return tles[0], tles[1]


class SatelliteCAMEnv(gym.Env):
    """
    Gymnasium environment for satellite collision avoidance maneuvers.
    Integrates real TLE data and poliastro orbital mechanics.
    
    State: [rel_pos_x, rel_pos_y, rel_pos_z, rel_vel_x, rel_vel_y, rel_vel_z, distance]
    Action: delta-V in radial, in-track, cross-track [km/s]
    """
    
    def __init__(
        self,
        scenario_file: Optional[str] = None,
        chaser_norad_id: Optional[int] = None,
        threat_norad_id: Optional[int] = None,
        use_real_orbital: bool = True,
        max_steps: int = 100,
        time_step: float = 60.0,
        collision_distance: float = 0.1,
        safe_distance: float = 1.5,
        max_deviation: float = 10.0
    ):
        super().__init__()
        
        self.scenario_file = scenario_file
        self.chaser_norad_id = chaser_norad_id
        self.threat_norad_id = threat_norad_id
        self.use_real_orbital = use_real_orbital and POLIASTRO_AVAILABLE
        self.max_steps = max_steps
        self.time_step = time_step
        self.collision_distance = collision_distance
        self.safe_distance = safe_distance
        self.max_deviation = max_deviation
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=-0.01, high=0.01, shape=(3,), dtype=np.float64
        )
        
        self.state = None
        self.steps = 0
        self.trajectory = []
        self.initial_distance = None
        self.previous_distance = None
        
        self.orbit_chaser = None
        self.orbit_threat = None
        self.orbit_chaser_original = None
        
        self.satellite_position = None
        self.satellite_velocity = None
        self.threat_position = None
        self.threat_velocity = None
        self.initial_position = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.steps = 0
        self.trajectory = []
        self.previous_distance = None
        
        if self.scenario_file and Path(self.scenario_file).exists():
            self._load_from_tle_file()
        elif self.chaser_norad_id and self.threat_norad_id:
            self._load_from_norad()
        else:
            if self.use_real_orbital:
                self._load_default()
            else:
                self._setup_simplified()
        
        rel_pos, rel_vel, distance = self._get_relative_state()
        self.initial_distance = distance
        self.state = np.concatenate([rel_pos, rel_vel, [distance]])
        
        info = {
            'initial_distance': float(distance),
            'scenario_type': 'real_orbital' if self.use_real_orbital else 'simplified',
            'chaser_id': self.chaser_norad_id,
            'threat_id': self.threat_norad_id
        }
        
        return self.state.astype(np.float64), info
    
    def _load_from_tle_file(self):
        """Load from TLE text file (Dr. Raouph's format)"""
        chaser_tle, threat_tle = parse_tle_file(self.scenario_file)
        
        self.chaser_norad_id = int(chaser_tle[1][2:7])
        self.threat_norad_id = int(threat_tle[1][2:7])
        
        self.orbit_chaser = create_orbit_from_tle(chaser_tle)
        self.orbit_threat = create_orbit_from_tle(threat_tle)
        self.orbit_chaser_original = self.orbit_chaser.copy()
        
        min_dist, tca = detect_close_approach(self.orbit_chaser, self.orbit_threat, 3600*24)
        if tca > 0:
            self.orbit_chaser = propagate_orbit(self.orbit_chaser, tca)
            self.orbit_threat = propagate_orbit(self.orbit_threat, tca)
    
    def _load_from_norad(self):
        """Load directly from Celestrak"""
        tle_chaser = load_tle_from_celestrak(self.chaser_norad_id)
        tle_threat = load_tle_from_celestrak(self.threat_norad_id)
        
        self.orbit_chaser = create_orbit_from_tle(tle_chaser)
        self.orbit_threat = create_orbit_from_tle(tle_threat)
        self.orbit_chaser_original = self.orbit_chaser.copy()
        
        min_dist, tca = detect_close_approach(self.orbit_chaser, self.orbit_threat, 3600*24)
        if tca > 0:
            self.orbit_chaser = propagate_orbit(self.orbit_chaser, tca)
            self.orbit_threat = propagate_orbit(self.orbit_threat, tca)
    
    def _load_default(self):
        """Default: Iridium-Cosmos collision"""
        self.chaser_norad_id = 24946
        self.threat_norad_id = 22675
        try:
            self._load_from_norad()
        except Exception:
            self.use_real_orbital = False
            self._setup_simplified()
    
    def _setup_simplified(self):
        """Simplified LVLH model"""
        self.satellite_position = np.random.uniform(-1, 1, size=3)
        self.satellite_velocity = np.random.uniform(-0.1, 0.1, size=3)
        self.initial_position = self.satellite_position.copy()
        self.threat_position = np.array([0.0, 0.0, 0.0])
        self.threat_velocity = np.random.uniform(-0.05, 0.05, size=3)
        self.orbit_chaser = None
        self.orbit_threat = None
    
    def _get_relative_state(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculate relative position, velocity, distance"""
        if self.use_real_orbital and self.orbit_chaser is not None:
            rel_pos, rel_vel, distance = calculate_relative_state(self.orbit_chaser, self.orbit_threat)
            return rel_pos, rel_vel, distance
        else:
            rel_pos = self.satellite_position - self.threat_position
            rel_vel = self.satellite_velocity - self.threat_velocity
            distance = np.linalg.norm(rel_pos)
            return rel_pos, rel_vel, distance
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.steps += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        if self.use_real_orbital and self.orbit_chaser is not None:
            self.orbit_chaser = apply_impulsive_maneuver(self.orbit_chaser, action)
            self.orbit_chaser = propagate_orbit(self.orbit_chaser, self.time_step)
            self.orbit_threat = propagate_orbit(self.orbit_threat, self.time_step)
        else:
            self.satellite_velocity = self.satellite_velocity + action
            self.satellite_position = self.satellite_position + self.satellite_velocity * self.time_step
            self.threat_position = self.threat_position + self.threat_velocity * self.time_step
        
        rel_pos, rel_vel, distance = self._get_relative_state()
        self.state = np.concatenate([rel_pos, rel_vel, [distance]])
        
        reward = self._calculate_reward(action, distance)
        terminated, truncated, info = self._check_termination(distance)
        
        self.trajectory.append({
            'rel_pos': rel_pos.copy(),
            'rel_vel': rel_vel.copy(),
            'distance': distance,
            'action': action.copy(),
            'delta_v_m_s': float(np.linalg.norm(action) * 1000),
            'reward': reward
        })
        
        info.update({
            'step': self.steps,
            'distance_km': float(distance),
            'fuel_used_m_s': float(np.linalg.norm(action) * 1000)
        })
        
        return self.state.astype(np.float64), reward, terminated, truncated, info
    
    def _calculate_reward(self, action: np.ndarray, distance: float) -> float:
        reward = 0.0
        
        if self.use_real_orbital and self.orbit_chaser_original is not None:
            orig_pos, _, _ = calculate_relative_state(self.orbit_chaser_original, self.orbit_threat)
            curr_pos, _, _ = self._get_relative_state()
            deviation = np.linalg.norm(curr_pos - orig_pos)
        else:
            deviation = np.linalg.norm(self.satellite_position - self.initial_position)
        
        reward -= deviation * 2.0
        reward -= np.linalg.norm(action) * 1000 * 0.5
        
        if distance < self.collision_distance:
            reward -= 100.0
        
        if distance > self.safe_distance:
            reward += 20.0
        
        if self.previous_distance is not None and distance > self.previous_distance:
            reward += 2.0
        
        self.previous_distance = distance
        return reward
    
    def _check_termination(self, distance: float) -> Tuple[bool, bool, Dict]:
        terminated = False
        truncated = False
        info = {}
        
        if distance < self.collision_distance * 0.5:
            terminated = True
            info['termination_reason'] = 'collision'
            info['success'] = False
        
        elif distance > self.safe_distance and self.steps > 10:
            terminated = True
            info['termination_reason'] = 'success'
            info['success'] = True
        
        if self.use_real_orbital and self.orbit_chaser_original is not None:
            orig_pos, _, _ = calculate_relative_state(self.orbit_chaser_original, self.orbit_threat)
            curr_pos, _, _ = self._get_relative_state()
            deviation = np.linalg.norm(curr_pos - orig_pos)
        else:
            deviation = np.linalg.norm(self.satellite_position - self.initial_position)
        
        if deviation > self.max_deviation:
            terminated = True
            info['termination_reason'] = 'max_deviation'
            info['success'] = False
        
        elif self.steps >= self.max_steps:
            truncated = True
            info['termination_reason'] = 'timeout'
            info['success'] = distance > self.safe_distance
        
        return terminated, truncated, info
    
    def render(self):
        if len(self.trajectory) == 0:
            print("No trajectory to render")
            return
        
        rel_positions = np.array([t['rel_pos'] for t in self.trajectory])
        
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(rel_positions[:, 0], rel_positions[:, 1], rel_positions[:, 2], 'b-')
        ax1.scatter(0, 0, 0, color='red', s=100, marker='x', label='Threat')
        ax1.set_xlabel('Radial [km]')
        ax1.set_ylabel('In-track [km]')
        ax1.set_zlabel('Cross-track [km]')
        ax1.set_title('Relative Trajectory (LVLH)')
        ax1.legend()
        
        ax2 = fig.add_subplot(132)
        distances = [t['distance'] for t in self.trajectory]
        ax2.plot(distances, 'g-')
        ax2.axhline(y=self.collision_distance, color='r', linestyle='--', label='Collision')
        ax2.axhline(y=self.safe_distance, color='g', linestyle='--', label='Safe')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Distance [km]')
        ax2.set_title('Distance to Threat')
        ax2.legend()
        ax2.grid(True)
        
        ax3 = fig.add_subplot(133)
        fuel = np.cumsum([t['delta_v_m_s'] for t in self.trajectory])
        ax3.plot(fuel, 'm-')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Cumulative Delta-V [m/s]')
        ax3.set_title('Fuel Consumption')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("SatelliteCAMEnv Test")
    print("=" * 60)
    
    print("\n[Test] Simplified Model")
    env = SatelliteCAMEnv(use_real_orbital=False)
    obs, info = env.reset(seed=42)
    print(f"Initial state: {obs}")
    
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: dist={info['distance_km']:.3f}km, reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"Finished: {info.get('termination_reason')}, Success: {info.get('success')}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.render()
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
    
    


        