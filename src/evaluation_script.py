"""
Evaluation script for normal vs adversarial conditions
Generates comparison metrics and tables
"""

import numpy as np
import gymnasium as gym
from typing import List, Dict, Tuple, Any
from stable_baselines3 import PPO

# Fixed import - removed relative import
try:
    from security_module import AdversarialWrapper
except ImportError:
    from .security_module import AdversarialWrapper


def evaluate_agent(model, env, n_episodes=50, adversarial=False):
    """
    Evaluate trained agent with proper metric tracking.
    """
    episodes_data = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        min_distance = float('inf')
        total_fuel = 0.0
        steps = 0
        
        initial_distance = info.get('distance_km', float('inf'))
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            current_distance = info.get('distance_km', float('inf'))
            min_distance = min(min_distance, current_distance)
            total_fuel += info.get('fuel_used_m_s', 0.0)
            episode_reward += reward
            steps += 1
            
            if terminated:
                break
        
        episodes_data.append({
            'success': info.get('success', False),
            'collision': info.get('collision', False),
            'min_distance': min_distance if min_distance != float('inf') else 0.0,
            'final_distance': current_distance if steps > 0 else initial_distance,
            'fuel_used': total_fuel,
            'reward': episode_reward,
            'steps': steps
        })
    
    # Aggregate metrics
    metrics = {
        'success_rate': np.mean([e['success'] for e in episodes_data]),
        'collision_rate': np.mean([e['collision'] for e in episodes_data]),
        'avg_min_distance': np.mean([e['min_distance'] for e in episodes_data]),
        'avg_fuel': np.mean([e['fuel_used'] for e in episodes_data]),
        'avg_reward': np.mean([e['reward'] for e in episodes_data]),
        'avg_steps': np.mean([e['steps'] for e in episodes_data]),
    }
    
    print(f"\n--- Evaluation Summary ---")
    print(f"Successes: {sum([e['success'] for e in episodes_data])}/{n_episodes}")
    print(f"Collisions: {sum([e['collision'] for e in episodes_data])}/{n_episodes}")
    
    return metrics, episodes_data


def compare_conditions(
    agent,
    env: AdversarialWrapper,
    n_episodes: int = 10,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Run full comparison between normal and adversarial conditions.
    """
    print(f"Running evaluation: {n_episodes} episodes per condition\n")
    
    # Normal evaluation
    print("NORMAL CONDITIONS")
    print("-" * 40)
    normal_metrics, normal_data = evaluate_agent(agent, env, n_episodes, adversarial=False)
    
    # Adversarial evaluation
    print("\nADVERSARIAL CONDITIONS")
    print("-" * 40)
    adv_metrics, adv_data = evaluate_agent(agent, env, n_episodes, adversarial=True)
    
    if verbose:
        print_comparison_table(normal_metrics, adv_metrics)
        print_performance_degradation(normal_metrics, adv_metrics)
        
    return normal_metrics, adv_metrics


def print_comparison_table(normal: Dict, adversarial: Dict) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print(f"{'METRIC':<25} {'NORMAL':<20} {'ADVERSARIAL':<20}")
    print("=" * 70)
    
    metrics_to_show = [
        ('Success Rate', 'success_rate', '.3f'),
        ('Collision Rate', 'collision_rate', '.3f'),
        ('Avg Fuel Used (m/s)', 'avg_fuel', '.3f'),
        ('Avg Min Distance (km)', 'avg_min_distance', '.2f'),
        ('Avg Reward', 'avg_reward', '.2f'),
    ]
    
    for name, key, fmt in metrics_to_show:
        norm_val = normal.get(key, 0)
        adv_val = adversarial.get(key, 0)
        print(f"{name:<25} {format(norm_val, fmt):<20} {format(adv_val, fmt):<20}")
        
    print("=" * 70)


def print_performance_degradation(normal: Dict, adversarial: Dict) -> None:
    """Calculate and print performance degradation percentages."""
    print("\nPERFORMANCE DEGRADATION")
    print("-" * 40)
    
    success_drop = (normal['success_rate'] - adversarial['success_rate']) * 100
    print(f"Success Rate Drop:     {success_drop:+.1f}%")
    
    if normal['avg_fuel'] > 0:
        fuel_increase = ((adversarial['avg_fuel'] - normal['avg_fuel']) / normal['avg_fuel']) * 100
        print(f"Fuel Consumption ↑:    {fuel_increase:+.1f}%")
    
    if normal['avg_min_distance'] > 0:
        distance_change = ((adversarial['avg_min_distance'] - normal['avg_min_distance']) / 
                          normal['avg_min_distance']) * 100
        print(f"Min Distance Change:   {distance_change:+.1f}%")


def generate_latex_table(normal: Dict, adversarial: Dict) -> str:
    """Generate LaTeX table for paper."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Performance Comparison: Normal vs Adversarial Conditions}
\label{tab:security_comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Normal} & \textbf{Adversarial} \\
\midrule
"""
    metrics = [
        ('Success Rate', 'success_rate', '.3f'),
        ('Collision Rate', 'collision_rate', '.3f'),
        ('Avg Fuel Used (m/s)', 'avg_fuel', '.3f'),
        ('Avg Min Distance (km)', 'avg_min_distance', '.2f'),
    ]
    
    for name, key, fmt in metrics:
        norm_val = format(normal.get(key, 0), fmt)
        adv_val = format(adversarial.get(key, 0), fmt)
        latex += f"{name} & {norm_val} & {adv_val} \\\\\n"
        
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


if __name__ == "__main__":
    # Demo with mock data
    print("DEMO: Mock evaluation results\n")
    
    mock_normal = {
        'success_rate': 0.92,
        'collision_rate': 0.08,
        'avg_fuel': 0.45,
        'avg_min_distance': 125.5,
        'avg_reward': -12.3,
    }
    
    mock_adversarial = {
        'success_rate': 0.65,
        'collision_rate': 0.35,
        'avg_fuel': 0.78,
        'avg_min_distance': 52.3,
        'avg_reward': -45.7,
    }
    
    print_comparison_table(mock_normal, mock_adversarial)
    print_performance_degradation(mock_normal, mock_adversarial)
    print("\nLaTeX Table:")
    print(generate_latex_table(mock_normal, mock_adversarial))