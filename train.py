"""
Training script for Satellite CAM with PPO
Supports parallel CPU training using SubprocVecEnv
Author: Pouya (Person 4) - Coordinator
"""

import os
import numpy as np
import json
from datetime import datetime
from typing import Dict, Optional
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Custom modules
from environment import SatelliteCAMEnv, make_parallel_envs
from security_module import AdversarialWrapper, make_secure_env

def make_env(scenario_file=None, use_real_orbital=True, rank=0, seed=42):
    def _init():
        if scenario_file and scenario_file.endswith('.json'):
            env = SatelliteCAMEnv(
                scenario_file=scenario_file,
                use_real_orbital=True,
                max_steps=100,
                time_step=60.0
            )
        else:
            env = SatelliteCAMEnv(
                scenario_file=scenario_file,
                use_real_orbital=use_real_orbital,
                max_steps=100,
                time_step=60.0
            )
        # ✅ Different seed for each parallel environment
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init

def train_agent(
    n_envs: int = 4,
    total_timesteps: int = 500_000,
    use_real_orbital: bool = True,  # Set True if poliastro available
    scenario_file: Optional[str] = None,
    save_dir: str = "./models",
    log_dir: str = "./logs",
    seed: int = 42
) -> Dict:
    """
    Train PPO agent with parallel CPU environments
    """
    print(f"Starting training with {n_envs} parallel environments")
    print(f"Device: CPU (parallel)")

    train_use_real = False  # Always simplified for training
    eval_use_real = False  # User choice for evaluation
    
    print(f"[IEEE Hybrid] Training: Simplified Model (forced)")
    
    # Create vectorized environment for training
    env = SubprocVecEnv([make_env(scenario_file, train_use_real, i, seed) for i in range(n_envs)])
    
    # Create evaluation environment (single, no parallel)
    
    eval_env = SatelliteCAMEnv(
        scenario_file=scenario_file,
        use_real_orbital=False,
        max_steps=100
    )
    eval_env = Monitor(eval_env) 
    
    print(f"Using real orbital: {eval_use_real}")

    
    # PPO hyperparameters optimized for satellite control
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger network for complex dynamics
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Encourage exploration
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device='cpu',  # Force CPU usage
        seed=seed
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=save_dir,
        name_prefix="ppo_satellite"
    )
    
    # Train
    print("Training started...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_satellite_final")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Training info
    training_info = {
        'algorithm': 'PPO',
        'n_envs': n_envs,
        'total_timesteps': total_timesteps,
        'final_model_path': final_model_path,
        'use_real_orbital': use_real_orbital,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_satellite_final")
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # ✅ CRITICAL: Close environments to prevent memory leak
    env.close()
    eval_env.close()
    print("Environments closed successfully")
    
    return training_info

def evaluate_security(
    model_path: str,
    n_episodes: int = 50,
    use_real_orbital: bool = False,
    scenario_file: Optional[str] = None,
    results_file: str = "security_results.json"
):
    """
    Evaluate trained agent under normal vs adversarial conditions
    """
    from evaluation_script import evaluate_agent, print_comparison_table, print_performance_degradation
    
    print("\n" + "="*60)
    print("SECURITY EVALUATION: Normal vs Adversarial")
    print("="*60)
    
    # Load model
    env = SatelliteCAMEnv(use_real_orbital=use_real_orbital, scenario_file=scenario_file)
    model = PPO.load(model_path, env=env)
    
    # Normal evaluation
    print("\n--- NORMAL CONDITIONS ---")
    normal_env = make_secure_env(
        SatelliteCAMEnv(use_real_orbital=use_real_orbital, scenario_file=scenario_file),
        adversarial=False
    )
    normal_metrics, normal_data = evaluate_agent(model, normal_env, n_episodes, adversarial=False)
    
    # Adversarial evaluation
    print("\n--- ADVERSARIAL CONDITIONS ---")
    adv_env = make_secure_env(
        SatelliteCAMEnv(use_real_orbital=use_real_orbital, scenario_file=scenario_file),
        adversarial=True,
        noise_sigma=5.0,
        attack_prob=0.3
    )
    adv_metrics, adv_data = evaluate_agent(model, adv_env, n_episodes, adversarial=True)
    
    # Print results
    print_comparison_table(normal_metrics, adv_metrics)
    print_performance_degradation(normal_metrics, adv_metrics)
    
    # Save results
    results = {
        'normal': normal_metrics,
        'adversarial': adv_metrics,
        'degradation': {
            'success_rate_drop': normal_metrics['success_rate'] - adv_metrics['success_rate'],
            'fuel_increase_pct': ((adv_metrics['avg_fuel'] - normal_metrics['avg_fuel']) / 
                                 normal_metrics['avg_fuel'] * 100) if normal_metrics['avg_fuel'] > 0 else 0
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate Satellite CAM RL agent')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both',
                       help='Mode: train, eval, or both')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments for training')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Total training timesteps')
    parser.add_argument('--scenario', type=str, default=None,
                       help='Path to scenario JSON file')
    parser.add_argument('--real-orbital', action='store_true',
                       help='Use real orbital mechanics (requires poliastro)')
    parser.add_argument('--model-path', type=str, default='./models/ppo_satellite_final',
                       help='Path to trained model for evaluation')
    parser.add_argument('--n-eval-episodes', type=int, default=50,
                       help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("Starting training...")
        train_info = train_agent(
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            use_real_orbital=args.real_orbital,
            scenario_file=args.scenario,
            save_dir="./models",
            log_dir="./logs"
        )
    
    if args.mode in ['eval', 'both']:
        print("Starting evaluation...")
        results = evaluate_security(
            model_path=args.model_path,
            n_episodes=args.n_eval_episodes,
            use_real_orbital=args.real_orbital,
            scenario_file=args.scenario
        )

if __name__ == "__main__":
    main()