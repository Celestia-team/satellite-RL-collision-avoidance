"""
Evaluation script for normal vs adversarial conditions
Generates comparison metrics and tables
"""

import numpy as np
import gymnasium as gym
from typing import List, Dict, Tuple, Any
from stable_baselines3 import PPO
from security_module import AdversarialWrapper


def evaluate_agent(
    agent,
    env: AdversarialWrapper,
    n_episodes: int = 10,
    adversarial: bool = False,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate agent in normal or adversarial conditions.
    
    Returns metrics: success_rate, avg_fuel, avg_min_distance, avg_reward
    """
    env.set_adversarial_mode(adversarial)
    
    episodes_data = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        episode_fuel = 0.0
        min_distance = float('inf')
        episode_reward = 0.0
        steps = 0
        
        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=deterministic)
            
            # Track fuel consumption (magnitude of delta-V)
            fuel_consumed = np.linalg.norm(action)
            episode_fuel += fuel_consumed
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Track minimum distance to threat
            if 'distance' in info:
                min_distance = min(min_distance, info['distance'])
            elif hasattr(env, 'current_distance'):
                min_distance = min(min_distance, env.current_distance)
                
        # Determine success (avoided collision and stayed close to original orbit)
        success = info.get('success', False) or (not info.get('collision', True))
        
        episodes_data.append({
            'success': success,
            'fuel': episode_fuel,
            'min_distance': min_distance if min_distance != float('inf') else 0,
            'reward': episode_reward,
            'steps': steps,
            'collision': info.get('collision', False)
        })
    
    # Calculate aggregate metrics
    metrics = {
        'success_rate': np.mean([ep['success'] for ep in episodes_data]),
        'avg_fuel': np.mean([ep['fuel'] for ep in episodes_data]),
        'avg_min_distance': np.mean([ep['min_distance'] for ep in episodes_data]),
        'avg_reward': np.mean([ep['reward'] for ep in episodes_data]),
        'collision_rate': np.mean([ep['collision'] for ep in episodes_data]),
        'std_fuel': np.std([ep['fuel'] for ep in episodes_data]),
        'std_distance': np.std([ep['min_distance'] for ep in episodes_data])
    }
    
    return metrics, episodes_data


def compare_conditions(
    agent,
    env: AdversarialWrapper,
    n_episodes: int = 10,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Run full comparison between normal and adversarial conditions.
    Returns both metric dictionaries and prints comparison table.
    """
    print(f"Running evaluation: {n_episodes} episodes per condition\n")
    
    # Normal evaluation
    print("🛡️  NORMAL CONDITIONS")
    print("-" * 40)
    normal_metrics, normal_data = evaluate_agent(agent, env, n_episodes, adversarial=False)
    
    # Adversarial evaluation
    print("\n⚠️  ADVERSARIAL CONDITIONS")
    print("-" * 40)
    adv_metrics, adv_data = evaluate_agent(agent, env, n_episodes, adversarial=True)
    
    # Print comparison table
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
        ('Avg Fuel Used', 'avg_fuel', '.3f'),
        ('Avg Min Distance (km)', 'avg_min_distance', '.2f'),
        ('Avg Reward', 'avg_reward', '.2f'),
        ('Fuel Std Dev', 'std_fuel', '.3f'),
        ('Distance Std Dev', 'std_distance', '.3f')
    ]
    
    for name, key, fmt in metrics_to_show:
        norm_val = normal.get(key, 0)
        adv_val = adversarial.get(key, 0)
        print(f"{name:<25} {norm_val:<20{fmt}} {adv_val:<20{fmt}}")
        
    print("=" * 70)


def print_performance_degradation(normal: Dict, adversarial: Dict) -> None:
    """Calculate and print performance degradation percentages."""
    print("\n📉 PERFORMANCE DEGRADATION")
    print("-" * 40)
    
    # Success rate drop (negative is bad)
    success_drop = (normal['success_rate'] - adversarial['success_rate']) * 100
    print(f"Success Rate Drop:     {success_drop:+.1f}%")
    
    # Fuel increase (positive is bad - more fuel used)
    if normal['avg_fuel'] > 0:
        fuel_increase = ((adversarial['avg_fuel'] - normal['avg_fuel']) / normal['avg_fuel']) * 100
        print(f"Fuel Consumption ↑:    {fuel_increase:+.1f}%")
    
    # Distance decrease (negative is bad - closer to collision)
    if normal['avg_min_distance'] > 0:
        distance_change = ((adversarial['avg_min_distance'] - normal['avg_min_distance']) / normal['avg_min_distance']) * 100
        print(f"Min Distance Change:   {distance_change:+.1f}%")
        
    print("-" * 40)


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
        ('Avg Fuel Used', 'avg_fuel', '.3f'),
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


def save_results_to_json(
    normal_metrics: Dict,
    adversarial_metrics: Dict,
    filename: str = 'security_evaluation_results.json'
) -> None:
    """Save evaluation results to JSON file."""
    import json
    
    results = {
        'normal': normal_metrics,
        'adversarial': adversarial_metrics,
        'degradation': {
            'success_rate_drop': normal_metrics['success_rate'] - adversarial_metrics['success_rate'],
            'fuel_increase_pct': ((adversarial_metrics['avg_fuel'] - normal_metrics['avg_fuel']) / normal_metrics['avg_fuel'] * 100) if normal_metrics['avg_fuel'] > 0 else 0
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Results saved to {filename}")


# Example usage for testing (without full environment)
def demo_evaluation():
    """Demo with mock data to show table format."""
    print("DEMO: Mock evaluation results\n")
    
    mock_normal = {
        'success_rate': 0.92,
        'collision_rate': 0.08,
        'avg_fuel': 0.45,
        'avg_min_distance': 125.5,
        'avg_reward': -12.3,
        'std_fuel': 0.12,
        'std_distance': 15.2
    }
    
    mock_adversarial = {
        'success_rate': 0.65,
        'collision_rate': 0.35,
        'avg_fuel': 0.78,
        'avg_min_distance': 52.3,
        'avg_reward': -45.7,
        'std_fuel': 0.25,
        'std_distance': 28.4
    }
    
    print_comparison_table(mock_normal, mock_adversarial)
    print_performance_degradation(mock_normal, mock_adversarial)
    print("\nLaTeX Table:")
    print(generate_latex_table(mock_normal, mock_adversarial))


if __name__ == "__main__":
    demo_evaluation()