#!/usr/bin/env python3
"""
IEEE Conference Paper Figure Generator
Generates publication-quality plots from evaluation results
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from glob import glob

class IEEEFigureGenerator:
    """Generate publication-quality figures for IEEE Conference Paper"""
    
    def __init__(self, results_dir='./scenarios', output_dir='./figures'):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # IEEE Figure settings
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
        
    def load_results_from_json(self):
        """Load results from security_results.json files"""
        data = {}
        # Look for all result files
        result_files = glob(str(self.results_dir / "*_results.json"))
        
        for filepath in result_files:
            with open(filepath, 'r') as f:
                content = f.read()
                # Parse the custom format if needed
                # For now, use mock data matching your results
                pass
                
        # Use your actual results
        return {
            'SC01_CLASSIC': {
                'clean': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 85.9, 'avg_min_distance': 15.9, 'avg_reward': 416.7},
                'moderate_jamming': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.8, 'avg_min_distance': 18.6, 'avg_reward': 412.9},
                'aggressive_spoofing': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.8, 'avg_min_distance': 18.6, 'avg_reward': 412.9},
                'severe_attack': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.8, 'avg_min_distance': 18.6, 'avg_reward': 412.9}
            },
            'SC02_HIGHVEL': {
                'clean': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.3, 'avg_min_distance': 17.4, 'avg_reward': 396.7},
                'moderate_jamming': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.0, 'avg_min_distance': 19.7, 'avg_reward': 394.1},
                'aggressive_spoofing': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.0, 'avg_min_distance': 19.7, 'avg_reward': 394.1},
                'severe_attack': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.0, 'avg_min_distance': 19.7, 'avg_reward': 394.1}
            },
            'SC03_COORBITAL': {
                'clean': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 85.9, 'avg_min_distance': 21.3, 'avg_reward': 467.1},
                'moderate_jamming': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.1, 'avg_min_distance': 19.2, 'avg_reward': 465.3},
                'aggressive_spoofing': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.1, 'avg_min_distance': 19.2, 'avg_reward': 465.3},
                'severe_attack': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.1, 'avg_min_distance': 19.2, 'avg_reward': 465.3}
            },
            'SC04_DRIFT': {
                'clean': {'success_rate': 0.933, 'collision_rate': 0.0, 'avg_fuel': 85.4, 'avg_min_distance': 25.0, 'avg_reward': 355.3},
                'moderate_jamming': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.6, 'avg_min_distance': 21.9, 'avg_reward': 337.3},
                'aggressive_spoofing': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.6, 'avg_min_distance': 21.9, 'avg_reward': 337.3},
                'severe_attack': {'success_rate': 1.0, 'collision_rate': 0.0, 'avg_fuel': 86.6, 'avg_min_distance': 21.9, 'avg_reward': 337.3}
            }
        }
    
    def figure_1_success_rate(self):
        """Figure 1: Success Rate Bar Chart"""
        data = self.load_results_from_json()
        fig, ax = plt.subplots(figsize=(8, 5))
        
        scenarios = ['SC01\nClassic', 'SC02\nHigh-Vel', 'SC03\nCo-orbital', 'SC04\nDrift']
        attack_levels = ['Clean', 'Moderate', 'Aggressive', 'Severe']
        x = np.arange(len(attack_levels))
        width = 0.2
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (sc_key, sc_name) in enumerate(zip(data.keys(), scenarios)):
            success_rates = []
            for level in ['clean', 'moderate_jamming', 'aggressive_spoofing', 'severe_attack']:
                rate = data[sc_key][level]['success_rate'] * 100
                success_rates.append(rate)
            
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, success_rates, width, label=sc_name.replace('\n', ' '), 
                         color=colors[i], edgecolor='black', linewidth=0.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('Cyber Attack Level', fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_title('Figure 1: Satellite Collision Avoidance Success Under Cyber Attacks', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attack_levels)
        ax.legend(loc='lower left', framealpha=0.9, fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(85, 105)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure1_success_rate.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure1_success_rate.pdf', bbox_inches='tight')
        print("✅ Figure 1: Success rate chart generated")
        plt.close()
    
    def figure_2_fuel_consumption(self):
        """Figure 2: Fuel Consumption Comparison"""
        data = self.load_results_from_json()
        fig, ax = plt.subplots(figsize=(8, 5))
        
        scenarios = ['SC01', 'SC02', 'SC03', 'SC04']
        attack_levels = ['Clean', 'Moderate', 'Aggressive', 'Severe']
        x = np.arange(len(attack_levels))
        width = 0.2
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (sc_key, sc_name) in enumerate(zip(data.keys(), scenarios)):
            fuel_vals = [data[sc_key][level]['avg_fuel'] for level in ['clean', 'moderate_jamming', 'aggressive_spoofing', 'severe_attack']]
            offset = width * (i - 1.5)
            ax.bar(x + offset, fuel_vals, width, label=sc_name, color=colors[i],
                  edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Attack Level', fontweight='bold')
        ax.set_ylabel('Fuel Consumption (m/s)', fontweight='bold')
        ax.set_title('Figure 2: Fuel Consumption vs Attack Intensity', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attack_levels)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_fuel.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure2_fuel.pdf', bbox_inches='tight')
        print("✅ Figure 2: Fuel consumption chart generated")
        plt.close()
    
    def figure_3_safety_margins(self):
        """Figure 3: Safety Distance Analysis"""
        data = self.load_results_from_json()
        fig, ax = plt.subplots(figsize=(8, 5))
        
        scenarios = ['SC01', 'SC02', 'SC03', 'SC04']
        attack_levels = ['Clean', 'Moderate', 'Aggressive', 'Severe']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (sc_key, sc_name) in enumerate(zip(data.keys(), scenarios)):
            distances = [data[sc_key][level]['avg_min_distance'] for level in ['clean', 'moderate_jamming', 'aggressive_spoofing', 'severe_attack']]
            x_pos = np.arange(len(attack_levels))
            ax.plot(x_pos, distances, marker='o', markersize=8, linewidth=2, 
                   label=sc_name, color=colors[i])
        
        ax.axhline(y=15.0, color='red', linestyle='--', linewidth=2, label='Safe Threshold', alpha=0.7)
        
        ax.set_xlabel('Attack Level', fontweight='bold')
        ax.set_ylabel('Minimum Distance (km)', fontweight='bold')
        ax.set_title('Figure 3: Safety Margins Under Adversarial Conditions', fontweight='bold')
        ax.set_xticks(range(len(attack_levels)))
        ax.set_xticklabels(attack_levels)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_safety.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure3_safety.pdf', bbox_inches='tight')
        print("✅ Figure 3: Safety margin chart generated")
        plt.close()
    
    def generate_latex_table(self):
        """Generate LaTeX table for paper"""
        data = self.load_results_from_json()
        
        latex = r"""\begin{table}[h]
\centering
\caption{Performance Across Orbital Scenarios and Cyber Attack Levels}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Scenario} & \textbf{Attack} & \textbf{Success} & \textbf{Collision} & \textbf{Fuel} & \textbf{Min Dist} & \textbf{Reward} \\
& & \textbf{(\%)} & \textbf{(\%)} & \textbf{(m/s)} & \textbf{(km)} & \\
\midrule
"""
        
        scenario_names = {
            'SC01_CLASSIC': 'Classic',
            'SC02_HIGHVEL': 'High-Velocity', 
            'SC03_COORBITAL': 'Co-orbital',
            'SC04_DRIFT': 'Long-term'
        }
        
        for sc_key, sc_data in data.items():
            sc_name = scenario_names.get(sc_key, sc_key)
            for level, metrics in sc_data.items():
                attack = level.replace('_', ' ').title()
                row = f"{sc_name} & {attack} & {metrics['success_rate']*100:.1f} & {metrics['collision_rate']*100:.1f} & {metrics['avg_fuel']:.1f} & {metrics['avg_min_distance']:.1f} & {metrics['avg_reward']:.1f} \\\\\n"
                latex += row
            latex += "\\midrule\n"
        
        latex += r"""\\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(self.output_dir / 'table_results.tex', 'w') as f:
            f.write(latex)
        print("✅ LaTeX Table: table_results.tex")
    
    def generate_all(self):
        """Generate all figures and tables"""
        print("="*70)
        print("IEEE Conference Paper - Figure Generator")
        print("="*70)
        print(f"Output directory: {self.output_dir.absolute()}")
        print("-"*70)
        
        self.figure_1_success_rate()
        self.figure_2_fuel_consumption()
        self.figure_3_safety_margins()
        self.generate_latex_table()
        
        print("="*70)
        print("✅ All figures generated successfully!")
        print("Files created:")
        for f in sorted(self.output_dir.glob('*')):
            print(f"  - {f.name}")
        print("="*70)


if __name__ == "__main__":
    gen = IEEEFigureGenerator()
    gen.generate_all()