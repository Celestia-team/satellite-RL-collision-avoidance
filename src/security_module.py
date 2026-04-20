"""
Cybersecurity module for Satellite Collision Avoidance
Adds adversarial attacks (noise/spoofing) and cyber risk penalties
Author: Maria Chowdhury (refactored by Celestia Team)
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Any


class AdversarialWrapper(gym.Wrapper):
    """
    Gymnasium wrapper to simulate cybersecurity attacks on satellite observations.
    
    Attacks simulated:
    - Gaussian noise: Simulates sensor jamming/interference
    - Position spoofing: Simulates GPS spoofing attack
    """
    
    def __init__(
        self,
        env: gym.Env,
        noise_sigma: float = 5.0,
        spoof_offset: np.ndarray = None,
        cyber_risk_weight: float = 0.01,
        attack_probability: float = 0.3
    ):
        """
        Args:
            env: Base Gymnasium environment (SatelliteCAMEnv)
            noise_sigma: Standard deviation for Gaussian noise (km)
            spoof_offset: Fixed offset added to position [x, y, z] (km)
            cyber_risk_weight: Weight for cyber penalty in reward
            attack_probability: Probability of attack in adversarial mode
        """
        super().__init__(env)
        self.noise_sigma = noise_sigma
        self.spoof_offset = spoof_offset if spoof_offset is not None else np.array([20.0, -15.0, 10.0])
        self.cyber_risk_weight = cyber_risk_weight
        self.attack_probability = attack_probability
        
        self.adversarial_mode = False
        self.current_observation = None
        
    def set_adversarial_mode(self, mode: bool = True) -> None:
        """Enable/disable adversarial attack mode."""
        self.adversarial_mode = mode
        
    def add_gaussian_noise(self, observation: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate sensor jamming."""
        noise = np.random.normal(0, self.noise_sigma, size=observation.shape)
        return observation + noise
        
    def add_position_spoofing(self, observation: np.ndarray) -> np.ndarray:
        """
        Add spoofing offset to position components (first 3 elements).
        Simulates GPS spoofing attack.
        """
        spoofed = observation.copy()
        if len(observation) >= 3:
            spoofed[0:3] += self.spoof_offset
        return spoofed
        
    def adversarial_attack(self, observation: np.ndarray) -> np.ndarray:
        """
        Combined attack: noise + spoofing with probability.
        """
        if np.random.random() > self.attack_probability:
            return observation
            
        # Step 1: Add Gaussian noise (jamming)
        attacked_obs = self.add_gaussian_noise(observation)
        # Step 2: Add position spoofing
        attacked_obs = self.add_position_spoofing(attacked_obs)
        
        return attacked_obs
    
        print(f"[ATTACK] Original obs: {observation[:3]}")
        print(f"[ATTACK] Spoofed obs: {spoofed[:3]}")
        print(f"[ATTACK] Noise applied: {np.linalg.norm(noise):.2f}")
        
    def calculate_cyber_risk_penalty(self, observation: np.ndarray) -> float:
        """
        Calculate penalty based on observation uncertainty.
        """
        if not self.adversarial_mode:
            return 0.0
            
        obs_magnitude = np.linalg.norm(observation)
        penalty = -self.cyber_risk_weight * obs_magnitude
        
        return penalty
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and apply attack if in adversarial mode."""
        obs, info = self.env.reset(**kwargs)
        
        self.current_observation = obs.copy()
        
        if self.adversarial_mode:
            obs = self.adversarial_attack(obs)
            info['adversarial_attack'] = not np.array_equal(obs, self.current_observation)
        else:
            info['adversarial_attack'] = False
            
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step with potential adversarial attack on observation.
        Modifies reward with cyber risk penalty.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_observation = obs.copy()
        
        # Apply adversarial attack if enabled
        if self.adversarial_mode:
            obs = self.adversarial_attack(obs)
            info['adversarial_attack'] = not np.array_equal(obs, self.current_observation)
        else:
            info['adversarial_attack'] = False
            
        # Add cyber risk penalty to reward
        cyber_penalty = self.calculate_cyber_risk_penalty(obs)
        reward += cyber_penalty
        info['cyber_penalty'] = cyber_penalty
        
        return obs, reward, terminated, truncated, info


class CyberRiskEvaluator:
    """
    Standalone evaluator for cybersecurity metrics.
    """
    
    def __init__(self, noise_sigma: float = 5.0, spoof_offset: np.ndarray = None):
        self.noise_sigma = noise_sigma
        self.spoof_offset = spoof_offset if spoof_offset is not None else np.array([20.0, -15.0, 10.0])
        
    def simulate_attack_effect(
        self,
        true_observation: np.ndarray,
        attack_type: str = 'both'
    ) -> Dict[str, np.ndarray]:
        """
        Simulate different attack types on a true observation.
        
        Args:
            true_observation: Clean observation vector
            attack_type: 'noise', 'spoofing', or 'both'
            
        Returns:
            Dictionary with original and attacked observations
        """
        results = {'original': true_observation.copy()}
        
        if attack_type in ['noise', 'both']:
            noise = np.random.normal(0, self.noise_sigma, size=true_observation.shape)
            results['with_noise'] = true_observation + noise
            
        if attack_type in ['spoofing', 'both']:
            spoofed = true_observation.copy()
            spoofed[0:3] += self.spoof_offset
            results['with_spoofing'] = spoofed
            
        if attack_type == 'both':
            combined = true_observation.copy()
            combined = combined + np.random.normal(0, self.noise_sigma, size=true_observation.shape)
            combined[0:3] += self.spoof_offset
            results['with_both'] = combined
            
        return results


# Helper function to wrap environment with security
def make_secure_env(base_env, adversarial=False, noise_sigma=5.0, attack_prob=0.3):
    """
    Factory function to create environment with security wrapper
    """
    wrapped = AdversarialWrapper(
        base_env,
        noise_sigma=noise_sigma,
        attack_probability=attack_prob
    )
    wrapped.set_adversarial_mode(adversarial)
    return wrapped