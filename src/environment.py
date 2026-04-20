"""
Satellite Collision Avoidance Environment (CAM)
Full integration with orbital_dynamics from Dr. Raouph
Author: Aparna (Person 1) - Refactored for parallel CPU training
"""

import matplotlib.pyplot as plt
import gymnasium as gym
import copy
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
        detect_close_approach,
        load_tle_from_file,
        TLEData
    )
    POLIASTRO_AVAILABLE = True
except ImportError as e:
    POLIASTRO_AVAILABLE = False
    print(f"Warning: orbital_dynamics not available ({e}), using simplified model")


class SatelliteCAMEnv(gym.Env):
    """
    Gymnasium environment for satellite collision avoidance maneuvers.
    Integrates real TLE data and poliastro orbital mechanics.
    
    State: [rel_pos_x, rel_pos_y, rel_pos_z, rel_vel_x, rel_vel_y, rel_vel_z, distance]
    Action: delta-V in radial, in-track, cross-track [km/s]
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        scenario_file: Optional[str] = None,
        chaser_norad_id: Optional[int] = None,
        threat_norad_id: Optional[int] = None,
        use_real_orbital: bool = False,
        max_steps: int = 50,
        time_step: float = 60.0,
        collision_distance: float = 0.1,
        safe_distance: float = 15.0,
        max_deviation: float = 100.0,
        render_mode: Optional[str] = None
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
        self.render_mode = render_mode
        
        # Action space: 3D delta-V in km/s (continuous)
        self.action_space = gym.spaces.Box(
            low=-0.01, high=0.01, shape=(3,), dtype=np.float32
        )
        
        # Observation space: [rel_pos(3), rel_vel(3), distance(1)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        # State variables
        self.state = None
        self.steps = 0
        self.trajectory = []
        self.initial_distance = None
        self.previous_distance = None
        
        # Orbital objects
        self.orbit_chaser = None
        self.orbit_threat = None
        self.orbit_chaser_original = None
        
        # Simplified model variables
        self.satellite_position = np.zeros(3)
        self.satellite_velocity = np.zeros(3)
        self.threat_position = np.zeros(3)
        self.threat_velocity = np.zeros(3)
        self.initial_position = np.zeros(3)

    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.steps = 0
        self.trajectory = []
        self.previous_distance = None
        
        # Load initial conditions
        if self.scenario_file and Path(self.scenario_file).exists():
            self._load_from_tle_file()
        elif self.chaser_norad_id and self.threat_norad_id:
            self._load_from_norad()
        else:
            if self.use_real_orbital:
                self._load_default()
            else:
                self._setup_simplified()
        
        # Get initial state
        rel_pos, rel_vel, distance = self._get_relative_state()
        self.initial_distance = distance
        self.state = np.concatenate([rel_pos, rel_vel, [distance]])
        
        info = {
            'initial_distance': float(distance),
            'distance_km': float(distance),
            'scenario_type': 'real_orbital' if self.use_real_orbital else 'simplified',
            'chaser_id': self.chaser_norad_id,
            'threat_id': self.threat_norad_id
        }
        
        return self.state.astype(np.float32), info
    
    # در environment.py، متد _load_from_tle_file را اصلاح کنید:

    def _load_from_tle_file(self):
        """Load from JSON scenario file (orbit or adversarial config)"""
        import json
        import os

        with open(self.scenario_file, 'r') as f:
            scenario = json.load(f)
    
        # Check if this is an adversarial config file (has base_scenario)
        if 'base_scenario' in scenario and 'attack_params' in scenario:
            # This is an adversarial config - load the base orbit scenario
            base_scenario_id = scenario['base_scenario']
            # Construct path to base scenario (same directory, _orbit.json suffix)
            base_dir = os.path.dirname(self.scenario_file)
            base_file = f"{base_scenario_id}_orbit.json"
            base_path = os.path.join(base_dir, base_file)
        
            print(f"[ENV] Adversarial config detected. Loading base scenario: {base_path}")
        
            # Load base scenario
            with open(base_path, 'r') as f:
                scenario = json.load(f)
            # Note: attack_params are ignored here - wrapper will handle them
    
        # Now scenario is always an orbit scenario
        self.chaser_norad_id = scenario['chaser_norad_id']
        self.threat_norad_id = scenario['threat_norad_id']
    
        # Update simulation params from scenario
        params = scenario.get('simulation_params', {})
        self.time_step = params.get('time_step_seconds', self.time_step)
        self.collision_distance = params.get('collision_distance_km', self.collision_distance)
        self.safe_distance = params.get('safe_distance_km', self.safe_distance)
        self.max_steps = params.get('max_steps', self.max_steps)
    
        # Load TLE data
        tle_path = scenario['tle_source_file']
        from orbital_dynamics import load_tle_from_file, create_orbit_from_tle, TLEData, detect_close_approach, propagate_orbit
    
        # Parse TLE file
        all_tles = self._parse_tle_for_norad(tle_path)
    
        chaser_tle = all_tles.get(self.chaser_norad_id)
        threat_tle = all_tles.get(self.threat_norad_id)
    
        if not chaser_tle or not threat_tle:
            raise ValueError(f"NORAD IDs {self.chaser_norad_id} or {self.threat_norad_id} not found in {tle_path}")
    
        self.orbit_chaser = create_orbit_from_tle(chaser_tle)
        self.orbit_threat = create_orbit_from_tle(threat_tle)
    
        # Hybrid Fallback: If real orbital distance is too large, use simplified
        rel_pos, rel_vel, distance = self._get_relative_state()
    
        if distance > 50.0 or distance == 0 or np.isnan(distance):
            print(f"[ENV] Warning: Large distance ({distance:.2f} km), using simplified model")
            self.use_real_orbital = False
            self._setup_simplified()


    def _parse_tle_for_norad(self, filepath):
        """Parse TLE file and return dict of {norad_id: TLEData}"""
        from orbital_dynamics import TLEData
        satellites = {}
        
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if line and not line.startswith('1 ') and not line.startswith('2 '):
                name = line
                if i + 2 < len(lines):
                    line1 = lines[i + 1]
                    line2 = lines[i + 2]
                    if line1.startswith('1 ') and line2.startswith('2 '):
                        try:
                            norad_id = int(line1[2:7].strip())
                            satellites[norad_id] = TLEData(
                                name=name, line1=line1, line2=line2, norad_id=norad_id
                            )
                        except ValueError:
                            pass
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        
        return satellites
    
    def _load_from_norad(self):
        """Load directly from Celestrak"""
        tle_chaser = load_tle_from_celestrak(self.chaser_norad_id)
        tle_threat = load_tle_from_celestrak(self.threat_norad_id)
        
        self.orbit_chaser = create_orbit_from_tle(tle_chaser)
        self.orbit_threat = create_orbit_from_tle(tle_threat)
        self.orbit_chaser_original = copy.deepcopy(self.orbit_chaser)
        
        min_dist, tca = detect_close_approach(self.orbit_chaser, self.orbit_threat, 3600*24)
        if tca > 0:
            self.orbit_chaser = propagate_orbit(self.orbit_chaser, tca)
            self.orbit_threat = propagate_orbit(self.orbit_threat, tca)
    
    def _load_default(self):
        """Default: Iridium-Cosmos collision scenario"""
        self.chaser_norad_id = 24946  # Iridium 33
        self.threat_norad_id = 22675  # Cosmos 2251 debris
        try:
            self._load_from_norad()
        except Exception as e:
            print(f"Failed to load default from Celestrak: {e}")
            self.use_real_orbital = False
            self._setup_simplified()
    
    def _setup_simplified(self):
        """Simplified LVLH model with proper initial separation"""
        # Ensure minimum separation - start 2-5 km away from threat at origin
        distance = np.random.uniform(2.0, 5.0)
        
        # Random direction with bias toward orbital plane
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        # Convert spherical to Cartesian
        x = distance * np.sin(phi) * np.cos(theta)
        y = distance * np.sin(phi) * np.sin(theta)
        z = distance * np.cos(phi) * 0.3
        
        self.satellite_position = np.array([x, y, z])
        self.satellite_velocity = np.random.uniform(-0.01, 0.01, size=3)
        self.initial_position = self.satellite_position.copy()
        
        # Threat at origin with realistic orbital drift
        self.threat_position = np.array([0.0, 0.0, 0.0])
        self.threat_velocity = np.array([
            np.random.uniform(-0.001, 0.001),
            np.random.uniform(0.003, 0.007),
            np.random.uniform(-0.001, 0.001)
        ])
        
        self.orbit_chaser = None
        self.orbit_threat = None
        
        initial_sep = np.linalg.norm(self.satellite_position - self.threat_position)
        print(f"[ENV] Simplified model initialized: separation={initial_sep:.2f} km")
    
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
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply maneuver and propagate
        if self.use_real_orbital and self.orbit_chaser is not None:
            self.orbit_chaser = apply_impulsive_maneuver(self.orbit_chaser, action)
            self.orbit_chaser = propagate_orbit(self.orbit_chaser, self.time_step)
            self.orbit_threat = propagate_orbit(self.orbit_threat, self.time_step)
        else:
            # Simplified physics
            self.satellite_velocity = self.satellite_velocity + action
            self.satellite_position = self.satellite_position + self.satellite_velocity * self.time_step
            self.threat_position = self.threat_position + self.threat_velocity * self.time_step
        
        # Get new state
        rel_pos, rel_vel, distance = self._get_relative_state()
        self.state = np.concatenate([rel_pos, rel_vel, [distance]])
        
        # Calculate reward
        reward = self._calculate_reward(action, distance)
        
        # Check termination
        terminated, truncated, info = self._check_termination(distance)
        
        # Store trajectory
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
        
        return self.state.astype(np.float32), float(reward), terminated, truncated, info
    
    def _calculate_reward(self, action: np.ndarray, distance: float) -> float:
        reward = 0.0

        # Heavy penalty for collision proximity
        if distance < self.collision_distance:
            reward -= 1000.0
        elif distance < self.safe_distance:
            reward -= (self.safe_distance - distance) * 10.0
        
        # Fuel cost (encourage efficiency)
        fuel_cost = np.linalg.norm(action) * 1000  # Convert to m/s
        reward -= fuel_cost * 0.01
        
        # Deviation penalty (stay close to original orbit)
        if self.use_real_orbital and self.orbit_chaser_original is not None:
            try:
                orig_pos, _, _ = calculate_relative_state(self.orbit_chaser_original, self.orbit_threat)
                curr_pos, _, _ = self._get_relative_state()
                deviation = np.linalg.norm(curr_pos - orig_pos)
            except:
                deviation = 0.0
        else:
            deviation = np.linalg.norm(self.satellite_position - self.initial_position)
        
        reward -= deviation * 0.05
        
        # Bonus for achieving safe distance
        if distance > self.safe_distance:
            reward += 100.0
        
        return reward
    
    def _check_termination(self, distance: float) -> Tuple[bool, bool, Dict]:
        terminated = False
        truncated = False
        info = {}
    
        # Collision check
        if distance < self.collision_distance:
            terminated = True
            info['termination_reason'] = 'collision'
            info['success'] = False
            info['collision'] = True
    
        # ✅ Success check - SIMPLIFIED (removed complex previous_distance check)
        elif distance > self.safe_distance and self.steps >= 5:
            terminated = True
            info['termination_reason'] = 'success'
            info['success'] = True
            info['collision'] = False
    
        # Max deviation check (relaxed)
        elif hasattr(self, 'satellite_position') and self.initial_position is not None:
            deviation = np.linalg.norm(self.satellite_position - self.initial_position)
            if deviation > self.max_deviation:
                terminated = True
                info['termination_reason'] = 'max_deviation'
                info['success'] = False
                info['collision'] = False
    
        # Timeout check
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            info['termination_reason'] = 'timeout'
            info['success'] = False
            info['collision'] = False
    
        if 'success' not in info:
            info['success'] = False
        if 'collision' not in info:
            info['collision'] = False
    
        return terminated, truncated, info
    
    def render(self):
        if len(self.trajectory) == 0:
            print("No trajectory to render")
            return
        
        rel_positions = np.array([t['rel_pos'] for t in self.trajectory])
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D Trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(rel_positions[:, 0], rel_positions[:, 1], rel_positions[:, 2], 'b-', label='Chaser')
        ax1.scatter(0, 0, 0, color='red', s=100, marker='x', label='Threat')
        ax1.set_xlabel('Radial [km]')
        ax1.set_ylabel('In-track [km]')
        ax1.set_zlabel('Cross-track [km]')
        ax1.set_title('Relative Trajectory (LVLH)')
        ax1.legend()
        
        # Distance over time
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
        
        # Fuel consumption
        ax3 = fig.add_subplot(133)
        fuel = np.cumsum([t['delta_v_m_s'] for t in self.trajectory])
        ax3.plot(fuel, 'm-')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Cumulative Delta-V [m/s]')
        ax3.set_title('Fuel Consumption')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()


def make_parallel_envs(n_envs: int = 4, **env_kwargs):
    """
    Create vectorized environments for parallel CPU training
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    
    def make_env():
        def _init():
            env = SatelliteCAMEnv(**env_kwargs)
            return env
        return _init
    
    # Use SubprocVecEnv for true multiprocessing on CPU
    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env()])
    
    return env


if __name__ == "__main__":
    print("=" * 60)
    print("SatelliteCAMEnv Test")
    print("=" * 60)
    
    # Test simplified model
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
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)