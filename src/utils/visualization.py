"""
Visualization Module for Bias Testing Results
Creates heatmaps and comparison charts
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime


# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try to import seaborn for better heatmaps
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class HeatmapVisualizer:
    """Creates heatmap visualizations of bias results."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {
            "colormap": "RdYlGn_r",
            "annot": True,
            "fmt": ".3f",
            "linewidths": 0.5
        }
    
    def create_bias_heatmap(self, metrics: Dict, output_path: str = None) -> str:
        """
        Create a heatmap of bias scores by category and method.
        
        Args:
            metrics: Dictionary of metrics by category and method
            output_path: Optional path to save the figure
            
        Returns:
            Path to saved figure or base64 encoded image
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib not available. Install with: pip install matplotlib"
        
        # Prepare data matrix
        categories = sorted(metrics.keys())
        methods = sorted(set(
            method 
            for cat_metrics in metrics.values() 
            for method in cat_metrics.keys()
        ))
        
        # Create bias score matrix
        bias_matrix = np.zeros((len(categories), len(methods)))
        
        for i, category in enumerate(categories):
            for j, method in enumerate(methods):
                if method in metrics[category]:
                    bias_matrix[i, j] = metrics[category][method].get("bias_score", 0)
                else:
                    bias_matrix[i, j] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(bias_matrix, cmap=self.config.get("colormap", "RdYlGn_r"), 
                       aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(categories)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Bias Score", rotation=-90, va="bottom")
        
        # Annotate cells
        if self.config.get("annot", True):
            fmt = self.config.get("fmt", ".3f")
            for i in range(len(categories)):
                for j in range(len(methods)):
                    if not np.isnan(bias_matrix[i, j]):
                        text = ax.text(j, i, format(bias_matrix[i, j], fmt),
                                       ha="center", va="center", 
                                       color="white" if bias_matrix[i, j] > 0.5 else "black")
        
        ax.set_title("Bias Scores by Category and Method")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            # Return as base64
            import base64
            from io import BytesIO
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_base64}"
    
    def create_accuracy_heatmap(self, metrics: Dict, output_path: str = None) -> str:
        """Create a heatmap of accuracy scores."""
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib not available"
        
        categories = sorted(metrics.keys())
        methods = sorted(set(
            method 
            for cat_metrics in metrics.values() 
            for method in cat_metrics.keys()
        ))
        
        accuracy_matrix = np.zeros((len(categories), len(methods)))
        
        for i, category in enumerate(categories):
            for j, method in enumerate(methods):
                if method in metrics[category]:
                    accuracy_matrix[i, j] = metrics[category][method].get("accuracy", 0)
                else:
                    accuracy_matrix[i, j] = np.nan
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(accuracy_matrix, cmap="RdYlGn", aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_yticklabels(categories)
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Accuracy", rotation=-90, va="bottom")
        
        if self.config.get("annot", True):
            for i in range(len(categories)):
                for j in range(len(methods)):
                    if not np.isnan(accuracy_matrix[i, j]):
                        text = ax.text(j, i, f"{accuracy_matrix[i, j]:.2f}",
                                       ha="center", va="center",
                                       color="white" if accuracy_matrix[i, j] < 0.5 else "black")
        
        ax.set_title("Accuracy by Category and Method")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            import base64
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"
    
    def create_comparison_chart(self, metrics: Dict, effectiveness: Dict, 
                                output_path: str = None) -> str:
        """Create comparison charts for methods."""
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib not available"
        
        methods = sorted(metrics.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bias score comparison
        bias_scores = [metrics[m].get("bias_score", 0) for m in methods]
        x = np.arange(len(methods))
        
        axes[0].bar(x, bias_scores, color=['red' if b > 0.3 else 'orange' if b > 0.1 else 'green' 
                                           for b in bias_scores])
        axes[0].set_xlabel('Method')
        axes[0].set_ylabel('Bias Score')
        axes[0].set_title('Bias Score by Method')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        axes[0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Low threshold')
        axes[0].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='High threshold')
        axes[0].legend()
        
        # Debias effectiveness
        if effectiveness:
            effect_methods = sorted(effectiveness.keys())
            reductions = [effectiveness[m].get('reduction_percentage', 0) for m in effect_methods]
            
            axes[1].bar(effect_methods, reductions, color='steelblue')
            axes[1].set_xlabel('Method')
            axes[1].set_ylabel('Bias Reduction (%)')
            axes[1].set_title('Debiasing Effectiveness')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            # Show accuracy comparison
            accuracies = [metrics[m].get('accuracy', 0) for m in methods]
            axes[1].bar(x, accuracies, color=['green' if a > 0.7 else 'orange' if a > 0.5 else 'red'
                                              for a in accuracies])
            axes[1].set_xlabel('Method')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy by Method')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            import base64
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"


class ComparisonVisualizer:
    """Creates comparison visualizations for different models and settings."""
    
    def __init__(self):
        pass
    
    def create_model_comparison(self, results_by_model: Dict[str, Dict], 
                               output_path: str = None) -> str:
        """Create comparison chart for different models."""
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib not available"
        
        models = list(results_by_model.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bias score comparison
        bias_scores = [results_by_model[m].get('bias_score', 0) for m in models]
        
        axes[0].bar(models, bias_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Bias Score')
        axes[0].set_title('Bias Score by Model')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Accuracy comparison
        accuracies = [results_by_model[m].get('accuracy', 0) for m in models]
        
        axes[1].bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy by Model')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            import base64
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"
    
    def create_parameter_comparison(self, fine_tune_results: List[Dict],
                                    output_path: str = None) -> str:
        """Create comparison for fine-tuning parameters."""
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib not available"
        
        # Group by parameter name
        params = {}
        for result in fine_tune_results:
            param_name = result.get('parameter_name', '')
            if param_name not in params:
                params[param_name] = {'values': [], 'bias_scores': [], 'accuracies': []}
            
            params[param_name]['values'].append(result.get('parameter_value', 0))
            params[param_name]['bias_scores'].append(result.get('bias_score', 0))
            params[param_name]['accuracies'].append(result.get('accuracy', 0))
        
        num_params = len(params)
        fig, axes = plt.subplots(1, num_params * 2, figsize=(4 * num_params * 2, 6))
        
        if num_params == 1:
            axes = [axes]
        
        for idx, (param_name, data) in enumerate(params.items()):
            values = data['values']
            
            # Bias score
            axes[idx].plot(values, data['bias_scores'], 'o-', label='Bias Score')
            axes[idx].set_xlabel(param_name)
            axes[idx].set_ylabel('Bias Score')
            axes[idx].set_title(f'{param_name} vs Bias')
            axes[idx].grid(True, alpha=0.3)
            
            # Accuracy
            if idx + num_params < len(axes):
                axes[idx + num_params].plot(values, data['accuracies'], 'o-', color='green', 
                                            label='Accuracy')
                axes[idx + num_params].set_xlabel(param_name)
                axes[idx + num_params].set_ylabel('Accuracy')
                axes[idx + num_params].set_title(f'{param_name} vs Accuracy')
                axes[idx + num_params].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            import base64
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"


def create_default_visualizations(results: List[Dict], output_dir: str = "results"):
    """
    Create default set of visualizations for results.
    
    Args:
        results: List of test results
        output_dir: Directory to save visualizations
    
    Returns:
        Dictionary of output paths
    """
    import os
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualizations.")
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    from bias_tester import BiasAnalyzer
    
    analyzer = BiasAnalyzer(results)
    metrics = analyzer.calculate_category_metrics()
    effectiveness = analyzer.calculate_debias_effectiveness()
    
    visualizer = HeatmapVisualizer()
    
    output_paths = {}
    
    # Create bias heatmap
    bias_path = os.path.join(output_dir, "bias_heatmap.png")
    visualizer.create_bias_heatmap(metrics, bias_path)
    output_paths['bias_heatmap'] = bias_path
    
    # Create accuracy heatmap
    accuracy_path = os.path.join(output_dir, "accuracy_heatmap.png")
    visualizer.create_accuracy_heatmap(metrics, accuracy_path)
    output_paths['accuracy_heatmap'] = accuracy_path
    
    # Create comparison chart
    comparison_path = os.path.join(output_dir, "comparison_chart.png")
    overall_metrics = analyzer.calculate_metrics()
    visualizer.create_comparison_chart(overall_metrics, effectiveness, comparison_path)
    output_paths['comparison_chart'] = comparison_path
    
    return output_paths
