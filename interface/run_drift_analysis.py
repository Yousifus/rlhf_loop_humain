#!/usr/bin/env python3
"""
Drift Analysis Runner

This script runs the drift monitoring analysis on historical meta-reflection data
and generates visualizations and reports to help identify model drift patterns.

Usage:
    python interface/run_drift_analysis.py [options]

Example:
    python interface/run_drift_analysis.py --visualization-mode detailed
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

# Import project modules
from utils.vote_predictor.drift_monitor import (
    run_drift_analysis, 
    DriftAnalysisConfig, 
    generate_visualizations
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, "models", "drift_analysis.log"))
    ]
)
logger = logging.getLogger(__name__)

def generate_report(drift_analysis, output_path):
    """
    Generate a detailed HTML report from drift analysis results.
    
    Args:
        drift_analysis: Results from the drift analysis
        output_path: Path to save the HTML report
    """
    try:
        # Create report template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RLHF Drift Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .warning {{ color: #cc0000; }}
                .success {{ color: #007700; }}
                .cluster {{ margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .viz-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                .viz-item {{ width: 48%; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>RLHF Drift Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="summary">
                    <h2>Summary</h2>
        """
        
        # Add summary section
        summary = drift_analysis.get("summary", {})
        time_drift_detected = summary.get("time_drift_detected", False)
        calibration_drift_detected = summary.get("calibration_drift_detected", False)
        
        drift_status_class = "warning" if time_drift_detected or calibration_drift_detected else "success"
        drift_status_text = "Drift Detected" if time_drift_detected or calibration_drift_detected else "No Significant Drift Detected"
        
        html_content += f"""
                    <p><strong class="{drift_status_class}">{drift_status_text}</strong></p>
                    <p>Total examples analyzed: {summary.get('total_examples', 0)}</p>
                    <p>Time-based drift detected: {'Yes' if time_drift_detected else 'No'}</p>
                    <p>Number of clusters: {summary.get('num_clusters', 0)}</p>
                    <p>Potential drift clusters: {len(summary.get('potential_drift_clusters', []))}</p>
                    <p>Calibration drift detected: {'Yes' if calibration_drift_detected else 'No'}</p>
                    <p>Calibration trend: {summary.get('calibration_trend', 'stable')}</p>
                </div>
        """
        
        # Add visualizations section
        html_content += """
                <h2>Visualizations</h2>
                <div class="viz-container">
        """
        
        # Add visualization images
        viz_paths = drift_analysis.get("visualization_paths", {})
        for viz_name, viz_path in viz_paths.items():
            if os.path.exists(viz_path):
                # Convert absolute path to relative path for HTML
                rel_path = os.path.relpath(viz_path, os.path.dirname(output_path))
                html_content += f"""
                    <div class="viz-item">
                        <h3>{viz_name.replace('_', ' ').title()}</h3>
                        <img src="{rel_path}" alt="{viz_name}" style="width: 100%;">
                    </div>
                """
        
        html_content += """
                </div>
        """
        
        # Add time analysis section
        html_content += """
                <h2>Time-Based Analysis</h2>
        """
        
        time_analysis = drift_analysis.get("time_analysis", {})
        windows = time_analysis.get("time_windows", [])
        
        if windows:
            html_content += """
                <table>
                    <tr>
                        <th>Period</th>
                        <th>Examples</th>
                        <th>Accuracy</th>
                        <th>Avg. Confidence</th>
                    </tr>
            """
            
            for window in windows:
                start = window.get("start_time", "").split("T")[0]
                end = window.get("end_time", "").split("T")[0]
                html_content += f"""
                    <tr>
                        <td>{start} to {end}</td>
                        <td>{window.get('example_count', 0)}</td>
                        <td>{window.get('accuracy', 0):.4f}</td>
                        <td>{window.get('avg_confidence', 0):.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            """
            
            # Add detected drifts
            drifts = time_analysis.get("detected_drifts", [])
            if drifts:
                html_content += """
                <h3>Detected Drifts</h3>
                <ul>
                """
                
                for drift in drifts:
                    html_content += f"""
                    <li class="warning">{drift.get('description', '')}</li>
                    """
                
                html_content += """
                </ul>
                """
        else:
            html_content += "<p>No time windows available for analysis.</p>"
        
        # Add clustering section
        html_content += """
                <h2>Cluster Analysis</h2>
        """
        
        clustering = drift_analysis.get("clustering_analysis", {})
        clusters = clustering.get("clusters", [])
        
        if clusters:
            html_content += f"""
                <p>Clustering algorithm: {clustering.get('algorithm', 'none')}</p>
                <h3>Identified Clusters</h3>
            """
            
            for cluster in clusters:
                cluster_class = "warning" if cluster.get("cluster_id") in summary.get("potential_drift_clusters", []) else ""
                html_content += f"""
                <div class="cluster {cluster_class}">
                    <h4>{cluster.get('description', '')}</h4>
                    <p>Size: {cluster.get('size', 0)} examples</p>
                    <p>Accuracy: {cluster.get('accuracy', 0):.4f}</p>
                    <p>Average confidence: {cluster.get('avg_confidence', 0):.4f}</p>
                </div>
                """
        else:
            html_content += "<p>No clusters identified.</p>"
        
        # Add calibration section
        html_content += """
                <h2>Confidence Calibration Analysis</h2>
        """
        
        calibration = drift_analysis.get("calibration_analysis", {})
        quartiles = calibration.get("quartile_calibration", [])
        
        if quartiles:
            html_content += """
                <table>
                    <tr>
                        <th>Quartile</th>
                        <th>Examples</th>
                        <th>Avg. Confidence</th>
                        <th>Accuracy</th>
                        <th>Calibration Error</th>
                    </tr>
            """
            
            for quartile in quartiles:
                html_content += f"""
                    <tr>
                        <td>{quartile.get('quartile', 0)}</td>
                        <td>{quartile.get('example_count', 0)}</td>
                        <td>{quartile.get('avg_confidence', 0):.4f}</td>
                        <td>{quartile.get('accuracy', 0):.4f}</td>
                        <td>{quartile.get('calibration_error', 0):.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            """
            
            trend = calibration.get("calibration_trend", "stable")
            trend_class = "warning" if trend == "worsening" else ("success" if trend == "improving" else "")
            
            html_content += f"""
                <p>Calibration trend: <span class="{trend_class}">{trend}</span></p>
            """
        else:
            html_content += "<p>No calibration data available.</p>"
        
        # Close HTML tags
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run drift analysis on RLHF system")
    
    parser.add_argument(
        "--reflection-path", 
        type=str, 
        default=os.path.join(project_root, "models", "meta_reflection_log.jsonl"),
        help="Path to the meta reflection log file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=os.path.join(project_root, "models", "drift_analysis"),
        help="Directory to save drift analysis results"
    )
    
    parser.add_argument(
        "--time-window-days", 
        type=int, 
        default=7,
        help="Size of time window in days for temporal analysis"
    )
    
    parser.add_argument(
        "--n-clusters", 
        type=int, 
        default=5,
        help="Number of clusters for KMeans analysis"
    )
    
    parser.add_argument(
        "--visualization-mode", 
        type=str, 
        choices=["basic", "detailed"],
        default="basic",
        help="Visualization mode (basic or detailed)"
    )
    
    parser.add_argument(
        "--generate-report", 
        action="store_true",
        help="Generate HTML report with analysis results"
    )
    
    parser.add_argument(
        "--confidence-drift-threshold", 
        type=float, 
        default=0.1,
        help="Threshold for detecting confidence calibration drift"
    )
    
    parser.add_argument(
        "--alert-accuracy-change-threshold", 
        type=float, 
        default=0.1,
        help="Threshold for alerting on accuracy changes"
    )
    
    args = parser.parse_args()
    
    # Configure drift analysis
    config = DriftAnalysisConfig(
        time_window_days=args.time_window_days,
        n_clusters=args.n_clusters,
        confidence_drift_threshold=args.confidence_drift_threshold,
        alert_accuracy_change_threshold=args.alert_accuracy_change_threshold
    )
    
    # Run drift analysis
    try:
        logger.info("Starting drift analysis")
        
        # Check if reflection log exists
        if not os.path.exists(args.reflection_path):
            logger.error(f"Reflection log not found at {args.reflection_path}")
            print(f"Error: Reflection log not found at {args.reflection_path}")
            return
        
        # Run analysis
        drift_analysis = run_drift_analysis(
            reflection_path=args.reflection_path,
            output_dir=args.output_dir,
            config=config
        )
        
        # Generate additional visualizations if detailed mode is selected
        if args.visualization_mode == "detailed":
            logger.info("Generating detailed visualizations")
            visualization_dir = os.path.join(args.output_dir, "visualizations")
            
            # Load reflections
            reflections = []
            with open(args.reflection_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        reflections.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
            
            # Generate additional visualizations
            additional_vizs = generate_visualizations(
                reflections, 
                drift_analysis.get("clustering_analysis", {}),
                visualization_dir
            )
            
            # Update visualization paths
            drift_analysis["visualization_paths"].update(additional_vizs)
        
        # Generate HTML report if requested
        if args.generate_report:
            logger.info("Generating HTML report")
            report_path = os.path.join(args.output_dir, "drift_report.html")
            generate_report(drift_analysis, report_path)
        
        # Output summary
        summary = drift_analysis.get("summary", {})
        print("\nDrift Analysis Summary:")
        print(f"Total examples analyzed: {summary.get('total_examples', 0)}")
        print(f"Time-based drift detected: {summary.get('time_drift_detected', False)}")
        print(f"Number of clusters: {summary.get('num_clusters', 0)}")
        print(f"Potential drift clusters: {summary.get('potential_drift_clusters', [])}")
        print(f"Calibration drift detected: {summary.get('calibration_drift_detected', False)}")
        print(f"Calibration trend: {summary.get('calibration_trend', 'stable')}")
        
        if args.generate_report:
            print(f"\nDetailed report saved to {report_path}")
        
        print(f"\nAnalysis saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error running drift analysis: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 