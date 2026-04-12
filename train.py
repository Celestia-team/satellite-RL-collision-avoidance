"""
Training and Evaluation Script for Satellite CAM with PPO
Integrates: Environment (Aparna) + Orbital Dynamics (Raouph) + Security (Maria)
Author: Pouya (Person 4) - Coordinator
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# RL libraries
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("ERROR: stable-baselines3 not installed. Run: pip install stable-baselines3")

# Local modules
from src.environment import SatelliteCAMEnv
from src.orbital_dynamics import (
    load_tle_from_celestrak, 
    create_orbit_from_tle,
    propagate_orbit,
    calculate_relative_state,
    detect_close_approach
)
from src.security_module import AdversarialWrapper
from src.evaluation_script import evaluate_agent, compare_conditions


# Configuration
CONFIG = {
    'scenario_file': 'scenarios/iridium-33-debris',
    'use_real_orbital': False,  # Start with simplified for faster training
    'max_steps': 100,
    'total_timesteps': 100000,
    'eval_episodes': 20,
    'seed': 42,
    
    # PPO hyperparameters
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    
    # Security evaluation
    'noise_levels': [0, 2, 5, 10],  # Sigma for Gaussian noise
    'attack_probability': 0.3,
    
    # Paths
    'models_dir': 'results/models',
    'logs_dir': 'results/logs',
    'plots_dir': 'results/plots',
}


def setup_directories():
    """Create necessary directories."""
    for dir_path in [CONFIG['models_dir'], CONFIG['logs_dir'], CONFIG['plots_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created")


def create_environment(use_real_orbital: bool = None, seed: int = None) -> SatelliteCAMEnv:
    """Create and wrap environment."""
    if use_real_orbital is None:
        use_real_orbital = CONFIG['use_real_orbital']
    
    env = SatelliteCAMEnv(
        scenario_file=CONFIG['scenario_file'] if Path(CONFIG['scenario_file']).exists() else None,
        use_real_orbital=use_real_orbital,
        max_steps=CONFIG['max_steps']
    )
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def train_agent():
    """Train PPO agent on collision avoidance task."""
    print("\n" + "="*70)
    print("PHASE 1: TRAINING")
    print("="*70)
    
    # Create environment
    print("\n[1] Creating environment...")
    env = create_environment(use_real_orbital=False, seed=CONFIG['seed'])
    
    # Check environment
    print("[2] Checking environment...")
    try:
        check_env(env)
        print("   ✓ Environment valid")
    except Exception as e:
        print(f"   ⚠ Environment check warning: {e}")
    
    # Wrap with Monitor for logging
    env = Monitor(env, CONFIG['logs_dir'] + "/training_monitor")
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    print("[3] Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=CONFIG['learning_rate'],
        n_steps=CONFIG['n_steps'],
        batch_size=CONFIG['batch_size'],
        n_epochs=CONFIG['n_epochs'],
        gamma=CONFIG['gamma'],
        gae_lambda=CONFIG['gae_lambda'],
        clip_range=CONFIG['clip_range'],
        ent_coef=CONFIG['ent_coef'],
        verbose=1,
        tensorboard_log=CONFIG['logs_dir'] + "/tensorboard/",
        seed=CONFIG['seed']
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=CONFIG['models_dir'],
        name_prefix="ppo_satellite_cam"
    )
    
    # Train
    print(f"[4] Training for {CONFIG['total_timesteps']} timesteps...")
    print("   (This may take several minutes...)")
    
    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{CONFIG['models_dir']}/ppo_satellite_cam_final.zip"
    model.save(final_model_path)
    print(f"   ✓ Model saved to {final_model_path}")
    
    # Save config
    config_path = f"{CONFIG['models_dir']}/training_config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"   ✓ Config saved to {config_path}")
    
    return model


def evaluate_normal(model):
    """Evaluate in normal (clean) conditions."""
    print("\n" + "="*70)
    print("PHASE 2A: NORMAL EVALUATION")
    print("="*70)
    
    env = create_environment(use_real_orbital=False)
    env = AdversarialWrapper(env, noise_sigma=0, attack_probability=0)
    
    print(f"\nEvaluating {CONFIG['eval_episodes']} episodes...")
    metrics, episodes = evaluate_agent(
        model, env, 
        n_episodes=CONFIG['eval_episodes'], 
        adversarial=False
    )
    
    print("\n--- Normal Conditions Results ---")
    print(f"Success Rate:      {metrics['success_rate']:.3f}")
    print(f"Collision Rate:      {metrics['collision_rate']:.3f}")
    print(f"Avg Fuel Used (m/s): {metrics['avg_fuel']:.3f}")
    print(f"Avg Min Distance:    {metrics['avg_min_distance']:.2f} km")
    print(f"Avg Reward:          {metrics['avg_reward']:.2f}")
    
    return metrics


def evaluate_adversarial(model, noise_sigma: float = 5.0):
    """Evaluate in adversarial (noisy) conditions."""
    print(f"\n{'='*70}")
    print(f"PHASE 2B: ADVERSARIAL EVALUATION (noise_sigma={noise_sigma})")
    print(f"{'='*70}")
    
    env = create_environment(use_real_orbital=False)
    env = AdversarialWrapper(
        env, 
        noise_sigma=noise_sigma,
        attack_probability=CONFIG['attack_probability'],
        cyber_risk_weight=0.01
    )
    
    print(f"\nEvaluating {CONFIG['eval_episodes']} episodes...")
    metrics, episodes = evaluate_agent(
        model, env,
        n_episodes=CONFIG['eval_episodes'],
        adversarial=True
    )
    
    print(f"\n--- Adversarial Conditions Results (σ={noise_sigma}) ---")
    print(f"Success Rate:      {metrics['success_rate']:.3f}")
    print(f"Collision Rate:      {metrics['collision_rate']:.3f}")
    print(f"Avg Fuel Used (m/s): {metrics['avg_fuel']:.3f}")
    print(f"Avg Min Distance:    {metrics['avg_min_distance']:.2f} km")
    print(f"Avg Reward:          {metrics['avg_reward']:.2f}")
    
    return metrics


def full_security_evaluation(model):
    """Evaluate across multiple noise levels."""
    print("\n" + "="*70)
    print("PHASE 3: FULL SECURITY EVALUATION")
    print("="*70)
    
    results = {}
    
    # Normal (baseline)
    results['normal'] = evaluate_normal(model)
    
    # Adversarial with different noise levels
    for noise in CONFIG['noise_levels'][1:]:  # Skip 0 (already done)
        results[f'adversarial_noise_{noise}'] = evaluate_adversarial(model, noise_sigma=noise)
    
    # Save results
    results_path = f"{CONFIG['logs_dir']}/security_evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        serializable_results = {}
        for key, val in results.items():
            serializable_results[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in val.items()
            }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    return results


def plot_comparison(results: Dict):
    """Generate comparison plots."""
    print("\n" + "="*70)
    print("PHASE 4: GENERATING PLOTS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    conditions = list(results.keys())
    success_rates = [results[k]['success_rate'] for k in conditions]
    collision_rates = [results[k]['collision_rate'] for k in conditions]
    fuel_usage = [results[k]['avg_fuel'] for k in conditions]
    min_distances = [results[k]['avg_min_distance'] for k in conditions]
    
    # Success rate
    axes[0, 0].bar(conditions, success_rates, color='green', alpha=0.7)
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate: Normal vs Adversarial')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Collision rate
    axes[0, 1].bar(conditions, collision_rates, color='red', alpha=0.7)
    axes[0, 1].set_ylabel('Collision Rate')
    axes[0, 1].set_title('Collision Rate: Normal vs Adversarial')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Fuel usage
    axes[1, 0].bar(conditions, fuel_usage, color='blue', alpha=0.7)
    axes[1, 0].set_ylabel('Avg Fuel Used (m/s)')
    axes[1, 0].set_title('Fuel Consumption: Normal vs Adversarial')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Min distance
    axes[1, 1].bar(conditions, min_distances, color='orange', alpha=0.7)
    axes[1, 1].set_ylabel('Avg Min Distance (km)')
    axes[1, 1].set_title('Minimum Approach Distance')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot_path = f"{CONFIG['plots_dir']}/security_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_path}")
    
    plt.close()
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    latex_path = f"{CONFIG['logs_dir']}/comparison_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to {latex_path}")


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table for paper."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Performance Comparison: Normal vs Adversarial Conditions}
\label{tab:security_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Condition} & \textbf{Success Rate} & \textbf{Collision Rate} & \textbf{Fuel (m/s)} & \textbf{Min Dist (km)} \\
\midrule
"""
    
    for condition, metrics in results.items():
        cond_name = condition.replace('_', ' ').title()
        latex += f"{cond_name} & "
        latex += f"{metrics['success_rate']:.3f} & "
        latex += f"{metrics['collision_rate']:.3f} & "
        latex += f"{metrics['avg_fuel']:.3f} & "
        latex += f"{metrics['avg_min_distance']:.2f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main():
    """Main execution pipeline."""
    print("="*70)
    print("SATELLITE COLLISION AVOIDANCE - RL TRAINING & EVALUATION")
    print("Celestia Team - LEO Satellite Systems with Cybersecurity")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    setup_directories()
    
    # Check dependencies
    if not STABLE_BASELINES_AVAILABLE:
        print("\nERROR: Please install stable-baselines3:")
        print("  pip install stable-baselines3")
        return
    
    # Phase 1: Train
    model = train_agent()
    
    # Phase 2 & 3: Evaluate
    results = full_security_evaluation(model)
    
    # Phase 4: Plot
    plot_comparison(results)
    
    # Summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nKey Results:")
    print(f"  Normal Success Rate:     {results['normal']['success_rate']:.3f}")
    if 'adversarial_noise_5' in results:
        print(f"  Adversarial (σ=5) Success: {results['adversarial_noise_5']['success_rate']:.3f}")
    print(f"\nAll outputs saved to:")
    print(f"  Models: {CONFIG['models_dir']}/")
    print(f"  Logs:   {CONFIG['logs_dir']}/")
    print(f"  Plots:  {CONFIG['plots_dir']}/")
    print("="*70)


if __name__ == "__main__":
    main()