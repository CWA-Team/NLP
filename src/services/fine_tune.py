"""
Fine-tuning Parameter Optimization Module
Tests and optimizes LLM parameters for bias reduction
"""

import itertools
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from api.llm_api import LLMAPI, get_llm
from services.bias_tester import BiasTester, BiasAnalyzer
from data.database import db
from utils.config import FINE_TUNE_PARAMS
from data.prompts_dataset import get_prompts_by_category


class ParameterOptimizer:
    """Optimizes LLM parameters for bias reduction."""
    
    def __init__(self, llm_provider: str = "openai", config: Dict = None):
        """
        Initialize parameter optimizer.
        
        Args:
            llm_provider: LLM provider to use
            config: Configuration dictionary
        """
        self.llm_provider = llm_provider
        self.config = config or {}
        self.results = []
    
    def test_parameter(self, param_name: str, param_value: Any, 
                      prompts: List[Dict], method: str = "baseline") -> Dict:
        """
        Test a single parameter value.
        
        Args:
            param_name: Name of the parameter
            param_value: Value to test
            prompts: List of prompts to test with
            method: Debiasing method to use
            
        Returns:
            Dictionary with results
        """
        # Create LLM with specific parameter
        test_config = self.config.copy()
        
        # Set the parameter to test
        if param_name == "temperature":
            test_config["temperature"] = param_value
        elif param_name == "max_tokens":
            test_config["max_tokens"] = param_value
        elif param_name == "top_p":
            test_config["top_p"] = param_value
        elif param_name == "presence_penalty":
            test_config["presence_penalty"] = param_value
        elif param_name == "frequency_penalty":
            test_config["frequency_penalty"] = param_value
        
        # Create tester with modified config
        tester = BiasTester(self.llm_provider, test_config)
        
        # Run tests
        results = []
        for prompt in prompts:
            if method == "baseline":
                result = tester.test_baseline(prompt)
            elif method == "explanation":
                result = tester.test_explanation_debiasing(prompt)
            elif method == "reprompting":
                result = tester.test_reprompting_debiasing(prompt)
            else:
                result = tester.test_baseline(prompt)
            
            result["category"] = prompt.get("category", "unknown")
            results.append(result)
            
            time.sleep(0.5)  # Rate limiting
        
        # Calculate metrics
        analyzer = BiasAnalyzer(results)
        metrics = analyzer.calculate_metrics()
        
        # Return results for this parameter
        return {
            "parameter_name": param_name,
            "parameter_value": param_value,
            "bias_score": metrics.get(method, {}).get("bias_score", 0),
            "accuracy": metrics.get(method, {}).get("accuracy", 0),
            "total_tests": metrics.get(method, {}).get("total_count", 0),
            "results": results
        }
    
    def optimize_single_parameter(self, param_name: str, param_values: List[Any],
                                  prompts: List[Dict], method: str = "baseline") -> Dict:
        """
        Optimize a single parameter.
        
        Args:
            param_name: Name of parameter to optimize
            param_values: List of values to test
            prompts: Prompts to test with
            method: Debiasing method
            
        Returns:
            Best parameter value and all results
        """
        print(f"Optimizing {param_name}...")
        
        all_results = []
        best_value = None
        best_bias = float('inf')
        
        for value in param_values:
            print(f"  Testing {param_name}={value}")
            
            result = self.test_parameter(param_name, value, prompts, method)
            all_results.append(result)
            
            # Track best
            if result["bias_score"] < best_bias:
                best_bias = result["bias_score"]
                best_value = value
            
            time.sleep(1)  # Delay between values
        
        return {
            "parameter": param_name,
            "best_value": best_value,
            "best_bias": best_bias,
            "all_results": all_results
        }
    
    def grid_search(self, prompts: List[Dict], parameters: Dict = None,
                   method: str = "baseline") -> Dict:
        """
        Perform grid search over parameter combinations.
        
        Args:
            prompts: Prompts to test with
            parameters: Parameter ranges (default: from config)
            method: Debiasing method
            
        Returns:
            Best parameter combination and results
        """
        if parameters is None:
            parameters = FINE_TUNE_PARAMS
        
        # For simplicity, test one parameter at a time
        best_params = {}
        best_bias = float('inf')
        
        for param_name, values in parameters.items():
            print(f"\nOptimizing {param_name}...")
            
            result = self.optimize_single_parameter(param_name, values, prompts, method)
            
            best_value = result["best_value"]
            best_params[param_name] = best_value
            
            if result["best_bias"] < best_bias:
                best_bias = result["best_bias"]
            
            print(f"  Best {param_name}: {best_value} (bias: {result['best_bias']:.4f})")
        
        return {
            "best_parameters": best_params,
            "best_bias_score": best_bias,
            "method": method
        }
    
    def find_optimal_settings(self, prompts: List[Dict], methods: List[str] = None) -> Dict:
        """
        Find optimal settings for all debiasing methods.
        
        Args:
            prompts: Prompts to test with
            methods: List of methods to optimize
            
        Returns:
            Optimal settings for each method
        """
        if methods is None:
            methods = ["baseline", "explanation", "reprompting"]
        
        results = {}
        
        for method in methods:
            print(f"\n{'='*50}")
            print(f"Optimizing for {method} method")
            print('='*50)
            
            # Test different temperature values (most impactful)
            temperatures = [0.1, 0.3, 0.5, 0.7]
            best_temp = None
            best_bias = float('inf')
            best_accuracy = 0
            
            for temp in temperatures:
                test_config = self.config.copy()
                test_config["temperature"] = temp
                
                tester = BiasTester(self.llm_provider, test_config)
                
                # Test on subset of prompts for speed
                test_prompts = prompts[:10]  # Use 10 prompts for quick testing
                
                results_list = []
                for prompt in test_prompts:
                    if method == "baseline":
                        result = tester.test_baseline(prompt)
                    elif method == "explanation":
                        result = tester.test_explanation_debiasing(prompt)
                    elif method == "reprompting":
                        result = tester.test_reprompting_debiasing(prompt)
                    else:
                        result = tester.test_baseline(prompt)
                    
                    results_list.append(result)
                    time.sleep(0.5)
                
                # Calculate bias
                analyzer = BiasAnalyzer(results_list)
                metrics = analyzer.calculate_metrics()
                bias = metrics.get("baseline", {}).get("bias_score", 0)
                accuracy = metrics.get("baseline", {}).get("accuracy", 0)
                
                print(f"  Temperature {temp}: bias={bias:.4f}, accuracy={accuracy:.4f}")
                
                # Find best (lowest bias, but maintain good accuracy)
                if bias < best_bias or (bias == best_bias and accuracy > best_accuracy):
                    best_bias = bias
                    best_accuracy = accuracy
                    best_temp = temp
            
            results[method] = {
                "best_temperature": best_temp,
                "best_bias": best_bias,
                "best_accuracy": best_accuracy,
                "config": {
                    "temperature": best_temp,
                    "max_tokens": 300,
                    "top_p": 0.9
                }
            }
            
            print(f"  Best for {method}: temp={best_temp}, bias={best_bias:.4f}")
        
        return results


class ParameterAnalyzer:
    """Analyzes parameter optimization results."""
    
    def __init__(self, results: List[Dict] = None):
        """
        Initialize analyzer.
        
        Args:
            results: List of parameter test results
        """
        self.results = results or []
    
    def find_optimal_parameters(self) -> Dict:
        """Find the optimal parameter values."""
        if not self.results:
            return {}
        
        # Group by parameter name
        params = {}
        for result in self.results:
            param_name = result.get("parameter_name", "")
            param_value = result.get("parameter_value", 0)
            bias_score = result.get("bias_score", 1)
            
            if param_name not in params:
                params[param_name] = []
            
            params[param_name].append({
                "value": param_value,
                "bias_score": bias_score
            })
        
        # Find best for each parameter
        optimal = {}
        for param_name, values in params.items():
            best = min(values, key=lambda x: x["bias_score"])
            optimal[param_name] = best["value"]
        
        return optimal
    
    def analyze_impact(self) -> Dict:
        """Analyze the impact of each parameter."""
        if not self.results:
            return {}
        
        # Group by parameter
        params = {}
        for result in self.results:
            param_name = result.get("parameter_name", "")
            if param_name not in params:
                params[param_name] = {
                    "values": [],
                    "bias_scores": [],
                    "accuracies": []
                }
            
            params[param_name]["values"].append(result.get("parameter_value", 0))
            params[param_name]["bias_scores"].append(result.get("bias_score", 0))
            params[param_name]["accuracies"].append(result.get("accuracy", 0))
        
        # Calculate impact (variance in bias scores)
        impact = {}
        for param_name, data in params.items():
            import numpy as np
            bias_variance = np.var(data["bias_scores"])
            impact[param_name] = {
                "variance": bias_variance,
                "range": max(data["bias_scores"]) - min(data["bias_scores"]),
                "values_tested": data["values"]
            }
        
        return impact
    
    def generate_recommendations(self) -> str:
        """Generate parameter recommendations."""
        optimal = self.find_optimal_parameters()
        impact = self.analyze_impact()
        
        recommendations = []
        recommendations.append("=" * 50)
        recommendations.append("PARAMETER OPTIMIZATION RECOMMENDATIONS")
        recommendations.append("=" * 50)
        recommendations.append("")
        
        recommendations.append("OPTIMAL PARAMETERS:")
        for param, value in optimal.items():
            recommendations.append(f"  {param}: {value}")
        
        recommendations.append("")
        recommendations.append("PARAMETER IMPACT ANALYSIS:")
        
        # Sort by impact
        sorted_impact = sorted(impact.items(), key=lambda x: x[1]["range"], reverse=True)
        
        for param_name, data in sorted_impact:
            recommendations.append(f"\n  {param_name}:")
            recommendations.append(f"    Impact (range): {data['range']:.4f}")
            recommendations.append(f"    Values tested: {data['values_tested']}")
        
        recommendations.append("")
        recommendations.append("IMPLEMENTATION RECOMMENDATIONS:")
        
        # Generate recommendations based on findings
        if "temperature" in optimal:
            temp = optimal["temperature"]
            if temp < 0.5:
                recommendations.append("  - Use lower temperature (0.1-0.3) for more consistent, less biased outputs")
            else:
                recommendations.append("  - Higher temperature can help explore diverse responses but may increase bias")
        
        recommendations.append("")
        recommendations.append("=" * 50)
        
        return "\n".join(recommendations)


def run_parameter_optimization(experiment_id: int, category: str = None,
                               methods: List[str] = None) -> Dict:
    """
    Run complete parameter optimization.
    
    Args:
        experiment_id: Database experiment ID
        category: Category to test (default: all)
        methods: Methods to optimize
        
    Returns:
        Optimization results
    """
    from utils.config import LLM_CONFIG
    from data.prompts_dataset import ALL_PROMPTS
    
    # Get prompts
    if category:
        prompts = get_prompts_by_category(category)
    else:
        prompts = ALL_PROMPTS
    
    # Use subset for faster testing
    test_prompts = prompts[:15]  # Use 15 prompts
    
    # Initialize optimizer
    optimizer = ParameterOptimizer("openai", LLM_CONFIG["openai"])
    
    # Run optimization
    optimal_settings = optimizer.find_optimal_settings(test_prompts, methods)
    
    # Analyze results
    analyzer = ParameterAnalyzer()
    recommendations = analyzer.generate_recommendations()
    
    # Save results to database
    for method, settings in optimal_settings.items():
        db.save_fine_tune_result(
            experiment_id,
            "optimal_temperature",
            settings.get("best_temperature", 0),
            settings.get("best_bias", 0),
            settings.get("best_accuracy", 0),
            0
        )
    
    return {
        "optimal_settings": optimal_settings,
        "recommendations": recommendations
    }
