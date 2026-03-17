"""
Bias Detection and Debiasing System - Main Application
Entry point for running bias tests, visualizations, and analysis
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils.config import LLM_CONFIG, BIAS_CATEGORIES, DEBIAS_METHODS
from data.database import db
from api.llm_api import get_llm
from services.bias_tester import BiasTester, BiasAnalyzer, save_results_to_db
from data.prompts_dataset import get_prompts_by_category, get_prompt_count, ALL_PROMPTS
from utils.visualization import HeatmapVisualizer, create_default_visualizations
from services.ai_summary import AISummaryGenerator, generate_full_report, ResultExporter
from services.fine_tune import ParameterOptimizer, run_parameter_optimization


class BiasDetectionSystem:
    """Main system for bias detection and debiasing."""
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize the bias detection system.
        
        Args:
            provider: LLM provider to use (openai or gemini)
        """
        self.provider = provider
        self.config = LLM_CONFIG.get(provider, LLM_CONFIG["openai"])
        self.tester = BiasTester(provider, self.config)
        self.experiment_id = None
    
    def create_experiment(self, name: str, description: str = "") -> int:
        """Create a new experiment in the database."""
        config = {
            "provider": self.provider,
            "model": self.config.get("model"),
            "categories": list(BIAS_CATEGORIES.keys()),
            "methods": list(DEBIAS_METHODS.keys())
        }
        
        self.experiment_id = db.create_experiment(name, description, config)
        db.update_experiment_status(self.experiment_id, "running")
        
        print(f"Created experiment #{self.experiment_id}: {name}")
        return self.experiment_id
    
    def run_tests(self, categories: list = None, methods: list = None,
                  save: bool = True) -> list:
        """
        Run bias tests.
        
        Args:
            categories: List of categories to test (default: all)
            methods: List of methods to test (default: all)
            save: Whether to save results to database
            
        Returns:
            List of test results
        """
        if categories is None:
            categories = list(BIAS_CATEGORIES.keys())
        
        if methods is None:
            methods = ["baseline", "explanation", "reprompting"]
        
        print(f"Running tests for categories: {categories}")
        print(f"Using methods: {methods}")
        
        all_results = []
        
        for category in categories:
            print(f"\n--- Testing {category} ---")
            prompts = get_prompts_by_category(category)
            
            for i, prompt in enumerate(prompts):
                print(f"  Prompt {i+1}/{len(prompts)}...")
                
                results = self.tester.run_full_test(prompt, methods)
                all_results.extend(results)
                
                # Save to database
                if save and self.experiment_id:
                    for result in results:
                        db.save_result(self.experiment_id, {
                            'bias_category': result.get('category', category),
                            'group_name': result.get('group_name', 'unknown'),
                            'method': result.get('method', 'unknown'),
                            'prompt': result.get('prompt', ''),
                            'response': result.get('response', ''),
                            'answer': result.get('answer', 'UNKNOWN'),
                            'is_biased': result.get('is_biased', False),
                            'bias_score': result.get('bias_score', 0.0),
                            'accuracy': 1.0 if result.get('answer') == result.get('correct_answer') else 0.0,
                            'response_time': result.get('response_time', 0.0)
                        })
        
        if self.experiment_id:
            db.update_experiment_status(self.experiment_id, "completed")
        
        return all_results
    
    def analyze_results(self, results: list) -> dict:
        """
        Analyze test results.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with analysis
        """
        analyzer = BiasAnalyzer(results)
        
        metrics = analyzer.calculate_metrics()
        category_metrics = analyzer.calculate_category_metrics()
        effectiveness = analyzer.calculate_debias_effectiveness()
        
        return {
            "overall_metrics": metrics,
            "category_metrics": category_metrics,
            "effectiveness": effectiveness,
            "summary": analyzer.generate_summary()
        }
    
    def generate_visualizations(self, results: list, output_dir: str = "results"):
        """
        Generate visualizations from results.
        
        Args:
            results: List of test results
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of output paths
        """
        print(f"Generating visualizations in {output_dir}...")
        
        analyzer = BiasAnalyzer(results)
        category_metrics = analyzer.calculate_category_metrics()
        overall_metrics = analyzer.calculate_metrics()
        effectiveness = analyzer.calculate_debias_effectiveness()
        
        visualizer = HeatmapVisualizer()
        
        output_paths = {}
        
        # Bias heatmap
        bias_path = os.path.join(output_dir, "bias_heatmap.png")
        visualizer.create_bias_heatmap(category_metrics, bias_path)
        output_paths["bias_heatmap"] = bias_path
        print(f"  Created: {bias_path}")
        
        # Accuracy heatmap
        accuracy_path = os.path.join(output_dir, "accuracy_heatmap.png")
        visualizer.create_accuracy_heatmap(category_metrics, accuracy_path)
        output_paths["accuracy_heatmap"] = accuracy_path
        print(f"  Created: {accuracy_path}")
        
        # Comparison chart
        comparison_path = os.path.join(output_dir, "comparison_chart.png")
        visualizer.create_comparison_chart(overall_metrics, effectiveness, comparison_path)
        output_paths["comparison_chart"] = comparison_path
        print(f"  Created: {comparison_path}")
        
        return output_paths
    
    def generate_ai_summary(self, results: list) -> str:
        """
        Generate AI-powered summary.
        
        Args:
            results: List of test results
            
        Returns:
            Summary text
        """
        print("Generating AI summary...")
        
        analyzer = BiasAnalyzer(results)
        metrics = analyzer.calculate_metrics()
        
        generator = AISummaryGenerator(self.provider, self.config)
        summary = generator.generate_summary(results, metrics)
        
        if self.experiment_id:
            db.save_summary(self.experiment_id, "ai_summary", summary, metrics)
        
        return summary
    
    def run_full_pipeline(self, categories: list = None, generate_reports: bool = True):
        """
        Run the full bias detection pipeline.
        
        Args:
            categories: Categories to test
            generate_reports: Whether to generate reports
            
        Returns:
            Dictionary with all results
        """
        print("=" * 60)
        print("BIAS DETECTION AND DEBIASING SYSTEM")
        print("=" * 60)
        
        # Create experiment
        self.create_experiment(
            name=f"Bias Test {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            description=f"Testing {categories or 'all categories'}"
        )
        
        # Run tests
        print("\n[1/5] Running bias tests...")
        results = self.run_tests(categories)
        print(f"  Completed {len(results)} tests")
        
        # Analyze results
        print("\n[2/5] Analyzing results...")
        analysis = self.analyze_results(results)
        print(analysis["summary"])
        
        # Generate visualizations
        print("\n[3/5] Generating visualizations...")
        output_dir = f"results_{self.experiment_id}"
        viz_paths = self.generate_visualizations(results, output_dir)
        
        # Generate AI summary
        print("\n[4/5] Generating AI summary...")
        ai_summary = self.generate_ai_summary(results)
        print(f"  Summary saved to database")
        
        # Generate reports
        print("\n[5/5] Generating reports...")
        if generate_reports:
            report_paths = generate_full_report(self.experiment_id, results, f"reports_{self.experiment_id}")
            print(f"  Reports saved to: reports_{self.experiment_id}")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
        return {
            "experiment_id": self.experiment_id,
            "results_count": len(results),
            "analysis": analysis,
            "visualizations": viz_paths
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Bias Detection and Debiasing System")
    
    parser.add_argument("--provider", "-p", default="openai",
                       choices=["openai", "gemini"],
                       help="LLM provider to use")
    parser.add_argument("--category", "-c", nargs="+",
                       choices=["religion", "socioeconomic", "gender"],
                       help="Categories to test")
    parser.add_argument("--methods", "-m", nargs="+",
                       choices=["baseline", "explanation", "reprompting", "chain_of_thought", "role_play"],
                       help="Debiasing methods to test")
    parser.add_argument("--output", "-o", default="results",
                       help="Output directory for results")
    parser.add_argument("--no-reports", action="store_true",
                       help="Skip generating reports")
    parser.add_argument("--test", action="store_true",
                       help="Run quick test with subset of prompts")
    
    args = parser.parse_args()
    
    # Initialize system
    system = BiasDetectionSystem(args.provider)
    
    if args.test:
        # Quick test with fewer prompts
        print("Running quick test...")
        
        # Test with first prompt from each category
        test_prompts = []
        for cat in ["religion", "socioeconomic", "gender"]:
            prompts = get_prompts_by_category(cat)
            if prompts:
                test_prompts.append(prompts[0])
        
        results = []
        for prompt in test_prompts:
            result = system.tester.run_full_test(prompt, args.methods or ["baseline", "explanation"])
            results.extend(result)
        
        analyzer = BiasAnalyzer(results)
        print(analyzer.generate_summary())
        
    else:
        # Full pipeline
        system.run_full_pipeline(
            categories=args.category,
            generate_reports=not args.no_reports
        )


if __name__ == "__main__":
    main()
