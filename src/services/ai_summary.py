"""
AI Summary Generation Module
Generates AI-powered summaries of bias testing results
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from api.llm_api import LLMAPI, get_llm
from data.database import db


class AISummaryGenerator:
    """Generates AI-powered summaries of bias testing results."""
    
    def __init__(self, llm_provider: str = "openai", config: Dict = None):
        """
        Initialize AI summary generator.
        
        Args:
            llm_provider: LLM provider to use
            config: Configuration dictionary
        """
        self.llm = get_llm(llm_provider, config)
    
    def generate_summary(self, results: List[Dict], metrics: Dict, 
                        category: str = "all") -> str:
        """
        Generate comprehensive summary of bias testing results.
        
        Args:
            results: List of test results
            metrics: Calculated metrics
            category: Category being tested
            
        Returns:
            Generated summary text
        """
        # Prepare summary data
        summary_data = self._prepare_summary_data(results, metrics, category)
        
        # Create prompt for summary generation
        prompt = self._create_summary_prompt(summary_data)
        
        # Generate summary
        response = self.llm.generate(prompt)
        
        return response.get("response", "Summary generation failed")
    
    def generate_comparison_summary(self, results_by_method: Dict[str, List[Dict]],
                                    metrics_by_method: Dict[str, Dict]) -> str:
        """
        Generate summary comparing different debiasing methods.
        
        Args:
            results_by_method: Results grouped by method
            metrics_by_method: Metrics grouped by method
            
        Returns:
            Generated comparison summary
        """
        # Prepare comparison data
        comparison_data = {
            "methods": [],
            "overall_bias": {},
            "best_performer": None,
            "recommendations": []
        }
        
        for method, metrics in metrics_by_method.items():
            comparison_data["methods"].append(method)
            comparison_data["overall_bias"][method] = metrics.get("bias_score", 0)
        
        # Find best performer
        if comparison_data["overall_bias"]:
            best = min(comparison_data["overall_bias"].items(), key=lambda x: x[1])
            comparison_data["best_performer"] = best[0]
        
        # Create prompt
        prompt = f"""As an AI expert in fairness and bias mitigation, analyze the following debiasing method comparison results:

Methods Tested: {', '.join(comparison_data['methods'])}

Overall Bias Scores:
{json.dumps(comparison_data['overall_bias'], indent=2)}

Best Performing Method: {comparison_data['best_performer']}

Please provide:
1. A brief analysis of which methods were most effective
2. Key insights about the bias patterns observed
3. Recommendations for which debiasing technique to use
4. Any concerns or limitations to note

Provide your analysis in a clear, professional format."""
        
        response = self.llm.generate(prompt)
        
        return response.get("response", "Comparison summary generation failed")
    
    def generate_category_summary(self, category: str, results: List[Dict],
                                  metrics: Dict) -> str:
        """
        Generate summary for a specific bias category.
        
        Args:
            category: Category name
            results: List of results for this category
            metrics: Calculated metrics
            
        Returns:
            Generated category summary
        """
        prompt = f"""As an AI expert in fairness and bias mitigation, analyze the following {category} bias testing results:

Test Results Summary:
- Total tests: {len(results)}
- Biased responses: {sum(1 for r in results if r.get('is_biased', False))}

Metrics by Method:
{json.dumps(metrics, indent=2)}

Please provide:
1. Analysis of the bias patterns specific to {category}
2. Which debiasing methods worked best for this category
3. Specific recommendations for reducing {category} bias
4. Any notable observations about the types of stereotypes encountered

Provide your analysis in a clear, professional format."""
        
        response = self.llm.generate(prompt)
        
        return response.get("response", "Category summary generation failed")
    
    def generate_recommendations(self, metrics: Dict, effectiveness: Dict) -> str:
        """
        Generate actionable recommendations based on results.
        
        Args:
            metrics: Overall metrics
            effectiveness: Debiasing effectiveness data
            
        Returns:
            Generated recommendations
        """
        # Find best method
        best_method = None
        best_reduction = 0
        
        for method, data in effectiveness.items():
            if data.get('reduction_percentage', 0) > best_reduction:
                best_reduction = data.get('reduction_percentage', 0)
                best_method = method
        
        prompt = f"""Based on the following bias testing results:

Overall Metrics:
{json.dumps(metrics, indent=2)}

Debiasing Effectiveness:
{json.dumps(effectiveness, indent=2)}

Best Method: {best_method} with {best_reduction:.1f}% bias reduction

Please provide:
1. Actionable recommendations for implementing debiasing in production systems
2. Specific parameter settings to optimize for bias reduction
3. Guidelines for when to use each debiasing technique
4. Warning signs that bias may be present in LLM outputs
5. Best practices for ongoing bias monitoring

Provide practical, implementation-focused recommendations."""
        
        response = self.llm.generate(prompt)
        
        return response.get("response", "Recommendations generation failed")
    
    def _prepare_summary_data(self, results: List[Dict], metrics: Dict, 
                              category: str) -> Dict:
        """Prepare data for summary generation."""
        total = len(results)
        biased = sum(1 for r in results if r.get('is_biased', False))
        
        # Group by method
        method_results = {}
        for r in results:
            method = r.get('method', 'unknown')
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(r)
        
        return {
            "category": category,
            "total_tests": total,
            "biased_count": biased,
            "bias_percentage": (biased / total * 100) if total > 0 else 0,
            "method_results": {m: len(rs) for m, rs in method_results.items()},
            "metrics": metrics
        }
    
    def _create_summary_prompt(self, data: Dict) -> str:
        """Create prompt for summary generation."""
        return f"""As an AI expert in fairness and bias mitigation, analyze the following bias testing results for the {data['category']} category:

Test Summary:
- Total tests conducted: {data['total_tests']}
- Biased responses: {data['biased_count']} ({data['bias_percentage']:.1f}%)
- Methods tested: {', '.join(data['method_results'].keys())}

Detailed Metrics:
{json.dumps(data['metrics'], indent=2)}

Please provide:
1. A professional summary of the key findings
2. Analysis of which debiasing techniques were most effective
3. Specific insights about the types of biases observed
4. Recommendations for reducing bias in {data['category']} contexts
5. Any important limitations or caveats to consider

Format your response with clear sections and bullet points where appropriate."""


class ResultExporter:
    """Exports results to various formats."""
    
    @staticmethod
    def export_to_json(results: List[Dict], filepath: str) -> bool:
        """Export results to JSON."""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    @staticmethod
    def export_to_csv(results: List[Dict], filepath: str) -> bool:
        """Export results to CSV."""
        try:
            import csv
            
            if not results:
                return False
            
            keys = results[0].keys()
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    @staticmethod
    def export_metrics_report(metrics: Dict, effectiveness: Dict, 
                            filepath: str) -> bool:
        """Export metrics as a formatted report."""
        try:
            report = []
            report.append("=" * 60)
            report.append("BIAS TESTING RESULTS REPORT")
            report.append("=" * 60)
            report.append("")
            
            report.append("OVERALL METRICS BY METHOD:")
            report.append("-" * 40)
            
            for method, data in metrics.items():
                report.append(f"\n{method.upper()}:")
                report.append(f"  Bias Score: {data.get('bias_score', 0):.4f}")
                report.append(f"  Accuracy: {data.get('accuracy', 0):.4f}")
                report.append(f"  Total Tests: {data.get('total_count', 0)}")
                report.append(f"  Biased Count: {data.get('biased_count', 0)}")
            
            report.append("")
            report.append("DEBIASING EFFECTIVENESS:")
            report.append("-" * 40)
            
            for method, data in effectiveness.items():
                report.append(f"\n{method}:")
                report.append(f"  Bias Reduction: {data.get('bias_reduction', 0):.4f}")
                report.append(f"  Reduction %: {data.get('reduction_percentage', 0):.1f}%")
                report.append(f"  New Bias Score: {data.get('new_bias_score', 0):.4f}")
            
            report.append("")
            report.append("=" * 60)
            report.append(f"Report generated: {datetime.now().isoformat()}")
            report.append("=" * 60)
            
            with open(filepath, 'w') as f:
                f.write('\n'.join(report))
            
            return True
        except Exception as e:
            print(f"Error exporting metrics report: {e}")
            return False


def generate_full_report(experiment_id: int, results: List[Dict],
                        output_dir: str = "reports") -> Dict[str, str]:
    """
    Generate full report with all summaries and visualizations.
    
    Args:
        experiment_id: Database experiment ID
        results: Test results
        output_dir: Directory to save reports
        
    Returns:
        Dictionary of output paths
    """
    import os
    from bias_tester import BiasAnalyzer
    
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = BiasAnalyzer(results)
    metrics = analyzer.calculate_metrics()
    category_metrics = analyzer.calculate_category_metrics()
    effectiveness = analyzer.calculate_debias_effectiveness()
    
    # Initialize generator
    generator = AISummaryGenerator()
    
    output_paths = {}
    
    # Generate overall summary
    summary = generator.generate_summary(results, metrics)
    summary_path = os.path.join(output_dir, f"summary_{experiment_id}.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    output_paths['summary'] = summary_path
    
    # Save to database
    db.save_summary(experiment_id, "overall", summary, metrics)
    
    # Generate category summaries
    for category in category_metrics.keys():
        cat_summary = generator.generate_category_summary(
            category, 
            [r for r in results if r.get('category') == category],
            category_metrics[category]
        )
        cat_path = os.path.join(output_dir, f"category_{category}_{experiment_id}.txt")
        with open(cat_path, 'w') as f:
            f.write(cat_summary)
        output_paths[f'category_{category}'] = cat_path
        
        db.save_summary(experiment_id, f"category_{category}", cat_summary, 
                       category_metrics[category])
    
    # Generate recommendations
    recommendations = generator.generate_recommendations(metrics, effectiveness)
    rec_path = os.path.join(output_dir, f"recommendations_{experiment_id}.txt")
    with open(rec_path, 'w') as f:
        f.write(recommendations)
    output_paths['recommendations'] = rec_path
    
    db.save_summary(experiment_id, "recommendations", recommendations, effectiveness)
    
    # Export metrics report
    exporter = ResultExporter()
    report_path = os.path.join(output_dir, f"metrics_report_{experiment_id}.txt")
    exporter.export_metrics_report(metrics, effectiveness, report_path)
    output_paths['metrics_report'] = report_path
    
    # Export JSON and CSV
    json_path = os.path.join(output_dir, f"results_{experiment_id}.json")
    exporter.export_to_json(results, json_path)
    output_paths['json'] = json_path
    
    csv_path = os.path.join(output_dir, f"results_{experiment_id}.csv")
    exporter.export_to_csv(results, csv_path)
    output_paths['csv'] = csv_path
    
    return output_paths
