"""Generate and save progression plots for each compound"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def generate_and_save_plots(user_data_path: str, compound: str) -> Optional[Path]:
    """
    Generate 4-chart progression visualization and save as PNG.
    
    Charts:
    1. Weight over time
    2. Load delta (session-to-session changes)
    3. Weight × Reps scatter (work capacity)
    4. Periodization cycles (deload detection)
    
    Args:
        user_data_path: Path to user data directory
        compound: Compound name (squat, bench_press, lat_pulldown, seated_row)
    
    Returns:
        Path to saved PNG, or None if failed
    """
    try:
        user_path = Path(user_data_path)
        
        # Determine username from path (users/{username}/)
        username = user_path.name
        csv_path = user_path / f"{username}_{compound}_history.csv"
        
        if not csv_path.exists():
            print(f"⚠ No history file found: {csv_path}")
            return None
        
        # Load history
        history = pd.read_csv(csv_path)
        
        if len(history) == 0:
            print(f"⚠ No data in {compound} history")
            return None
        
        # Add periodization features if not present
        if 'load_delta' not in history.columns:
            history['load_delta'] = history['weight'].diff()
        
        if 'is_deload' not in history.columns:
            # Detect deloads: 15% weight drop
            history['is_deload'] = (history['weight'].diff() / history['weight'].shift()) <= -0.15
        
        if 'weeks_in_cycle' not in history.columns:
            # Count weeks since deload
            hist_copy = history.copy()
            cycle_num = 0
            weeks_in = 0
            cycles = []
            weeks_list = []
            
            for idx, row in hist_copy.iterrows():
                if idx == 0:
                    weeks_in = 1
                elif row.get('is_deload', False):
                    cycle_num += 1
                    weeks_in = 1
                else:
                    weeks_in += 1
                cycles.append(cycle_num)
                weeks_list.append(weeks_in)
            
            history['cycle_number'] = cycles
            history['weeks_in_cycle'] = weeks_list
        
        # Get stats
        max_weight = history['weight'].max()
        last_weight = history['weight'].iloc[-1]
        last_reps = history['reps'].iloc[-1]
        last_rpe = history['rpe'].iloc[-1]
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{compound.replace('_', ' ').title()} Progression", fontsize=16, fontweight='bold')
        
        # Plot 1: Weight over time
        ax = axes[0, 0]
        ax.plot(range(len(history)), history['weight'], marker='o', markersize=3, label='Weight', alpha=0.7, color='blue')
        ax.scatter(len(history)-1, last_weight, color='darkblue', s=100, zorder=5, label='Current', edgecolors='black', linewidth=1)
        ax.axhline(max_weight, color='green', linestyle='--', alpha=0.5, label=f'Max: {max_weight:.1f} lbs')
        ax.set_xlabel('Session')
        ax.set_ylabel('Weight (lbs)')
        ax.set_title('Weight Progression Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Load delta distribution
        ax = axes[0, 1]
        deltas = history['load_delta'].dropna()
        if len(deltas) > 0:
            colors = ['green' if d >= 0 else 'red' for d in deltas]
            ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            ax.set_xlabel('Session (chronological)')
            ax.set_ylabel('load_delta (lbs)')
            ax.set_title('Load Changes by Session')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Reps vs Weight scatter (colored by time)
        ax = axes[1, 0]
        scatter = ax.scatter(history['weight'], history['reps'], c=range(len(history)), 
                            cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Weight (lbs)')
        ax.set_ylabel('Reps')
        ax.set_title('Weight × Reps (colored by time)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Session #')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Cycle detection (last 50 sessions)
        ax = axes[1, 1]
        tail_n = min(50, len(history))
        tail_data = history.tail(tail_n).reset_index(drop=True)
        colors = ['red' if d else 'blue' for d in tail_data.get('is_deload', [False]*len(tail_data))]
        ax.bar(range(len(tail_data)), tail_data['weeks_in_cycle'], color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Session (last 50)')
        ax.set_ylabel('Weeks in Cycle')
        ax.set_title('Periodization Cycles (red=deload, blue=climb)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save to plots directory
        plots_dir = user_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_path = plots_dir / f"{compound}_progression.png"
        fig.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return plot_path
    
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
        return None


def open_plot(plot_path: Optional[Path]) -> bool:
    """
    Open plot image in default viewer.
    
    Args:
        plot_path: Path to PNG file
    
    Returns:
        True if successful, False otherwise
    """
    if plot_path is None or not plot_path.exists():
        return False
    
    try:
        import os
        import platform
        
        if platform.system() == 'Darwin':  # macOS
            os.system(f'open "{plot_path}"')
        elif platform.system() == 'Windows':
            os.startfile(plot_path)
        else:  # Linux
            os.system(f'xdg-open "{plot_path}"')
        
        return True
    except Exception as e:
        print(f"✗ Error opening plot: {e}")
        return False
