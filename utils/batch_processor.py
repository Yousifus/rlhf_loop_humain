import os
import json
import time
import csv
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

from utils.completions import generate_completions

class BatchProcessor:
    """
    Class for batch processing multiple prompts with DeepSeek API or simulation.
    Handles parallel processing, error recovery, and result aggregation.
    """
    
    def __init__(
        self,
        output_dir: str = "data/batch_results",
        max_workers: int = 3,
        max_retries: int = 2,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256
    ):
        """Initialize the batch processor with configuration settings."""
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize batch run timestamp for output filenames
        self.batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Results tracking
        self.successful_items = 0
        self.failed_items = 0
        self.total_tokens = 0
        self.total_cost = 0
    
    def process_single_prompt(self, prompt: str, prompt_id: Optional[str] = None) -> Dict:
        """Process a single prompt and handle retries."""
        if not prompt_id:
            prompt_id = f"prompt_{hash(prompt) % 10000}"
            
        try:
            result = generate_completions(
                prompt=prompt,
                n_completions=2,  # Default to 2 completions per prompt
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                calculate_cost=True,
                fallback_on_error=True,
                max_retries=self.max_retries
            )
            
            # Add prompt_id to the result
            result["prompt_id"] = prompt_id
            
            return {
                "status": "success",
                "prompt_id": prompt_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "prompt_id": prompt_id,
                "error": str(e)
            }
    
    def process_batch(self, prompts: List[Dict]) -> Dict:
        """
        Process a batch of prompts in parallel.
        
        Args:
            prompts: List of dictionaries with "prompt" and optional "prompt_id" keys
            
        Returns:
            Dictionary with results and summary statistics
        """
        start_time = time.time()
        results = []
        
        # Create progress bar
        with tqdm(total=len(prompts), desc="Processing prompts") as pbar:
            # Process prompts in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_prompt = {
                    executor.submit(
                        self.process_single_prompt, 
                        item["prompt"], 
                        item.get("prompt_id", None)
                    ): item for item in prompts
                }
                
                # Process results as they complete
                for future in as_completed(future_to_prompt):
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        self.successful_items += 1
                        completion_result = result["result"]
                        self.total_tokens += completion_result["aggregated_usage"]["total_tokens"]
                        self.total_cost += completion_result.get("estimated_cost", 0)
                    else:
                        self.failed_items += 1
                    
                    pbar.update(1)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Save results to JSON file
        results_filename = f"{self.output_dir}/batch_results_{self.batch_timestamp}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Save results to CSV for easier analysis
        csv_filename = f"{self.output_dir}/batch_results_{self.batch_timestamp}.csv"
        self._save_results_to_csv(results, csv_filename)
            
        # Prepare summary stats
        summary = {
            "timestamp": self.batch_timestamp,
            "total_items": len(prompts),
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": round(self.successful_items / len(prompts) * 100, 2),
            "total_tokens": self.total_tokens,
            "estimated_cost": self.total_cost,
            "duration_seconds": round(duration, 2),
            "results_file": results_filename,
            "csv_file": csv_filename
        }
        
        return {
            "summary": summary,
            "results": results
        }
    
    def _save_results_to_csv(self, results: List[Dict], filename: str) -> None:
        """Save results to CSV file for easier analysis."""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['prompt_id', 'status', 'prompt', 'completion_1', 'completion_2', 
                         'tokens', 'cost', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for item in results:
                row = {
                    'prompt_id': item['prompt_id'],
                    'status': item['status'],
                    'error': item.get('error', '')
                }
                
                if item['status'] == 'success':
                    row['prompt'] = item['result']['prompt']
                    completions = item['result']['completions']
                    row['completion_1'] = completions[0] if len(completions) > 0 else ''
                    row['completion_2'] = completions[1] if len(completions) > 1 else ''
                    row['tokens'] = item['result']['aggregated_usage']['total_tokens']
                    row['cost'] = item['result'].get('estimated_cost', 0)
                
                writer.writerow(row)

# Example usage when run directly
if __name__ == "__main__":
    # Sample prompts for testing
    test_prompts = [
        {"prompt": "What is reinforcement learning?", "prompt_id": "rl_basics_1"},
        {"prompt": "Explain the concept of reward functions in RL.", "prompt_id": "rl_basics_2"},
        {"prompt": "How does RLHF improve AI alignment?", "prompt_id": "rlhf_1"}
    ]
    
    processor = BatchProcessor(
        max_workers=2,
        temperature=0.8,
        max_tokens=200
    )
    
    print(f"Starting batch processing of {len(test_prompts)} prompts...")
    batch_results = processor.process_batch(test_prompts)
    
    print("\nBatch processing complete!")
    print(f"Success rate: {batch_results['summary']['success_rate']}%")
    print(f"Total tokens: {batch_results['summary']['total_tokens']}")
    print(f"Results saved to: {batch_results['summary']['results_file']}")
    print(f"CSV export: {batch_results['summary']['csv_file']}") 