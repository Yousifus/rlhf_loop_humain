# Batch Processing Documentation

This document explains how to use the batch processing functionality in the RLHF Loop system to generate completions for multiple prompts efficiently.

## Overview

The batch processor allows you to:

1. Process multiple prompts in parallel
2. Track processing progress in real-time
3. Handle errors gracefully with automatic retry mechanisms
4. Generate detailed reports and exports
5. Calculate token usage and estimated costs

## Prerequisites

Before using the batch processor, ensure:

1. You have set up your DeepSeek API key (if using DeepSeek)
2. Python dependencies are installed (`tqdm`, etc.)
3. You have created a JSON file with your prompts

## Prompt File Format

The batch processor accepts JSON files with prompts in the following formats:

### Format 1: List of Dictionaries

```json
[
  {
    "prompt_id": "unique_id_1",
    "prompt": "Your prompt text here"
  },
  {
    "prompt_id": "unique_id_2", 
    "prompt": "Another prompt text"
  }
]
```

### Format 2: Dictionary with Prompts List

```json
{
  "batch_name": "My Batch",
  "description": "Optional description",
  "prompts": [
    {
      "prompt_id": "unique_id_1",
      "prompt": "Your prompt text here"
    },
    {
      "prompt_id": "unique_id_2",
      "prompt": "Another prompt text" 
    }
  ],
  "metadata": {
    "created": "2023-10-05",
    "author": "Your Name",
    "version": "1.0.0"
  }
}
```

### Format 3: Simple List of Strings

```json
[
  "Your prompt text here",
  "Another prompt text"
]
```

## Running Batch Processing

### Basic Usage

To run batch processing with default settings:

```powershell
.\run_batch_processor.ps1 -InputFile prompts/my_prompts.json
```

### Advanced Usage

```powershell
.\run_batch_processor.ps1 `
  -InputFile prompts/my_prompts.json `
  -MaxWorkers 4 `
  -Temperature 0.8 `
  -TopP 0.9 `
  -MaxTokens 300 `
  -OutputDir "data/custom_results"
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| InputFile | Path to the JSON prompts file | (required) |
| MaxWorkers | Number of parallel workers | 3 |
| Temperature | Model temperature | 0.7 |
| TopP | Top-p sampling value | 0.9 |
| MaxTokens | Maximum tokens per completion | 256 |
| OutputDir | Directory for output files | data/batch_results |

## Output Files

The batch processor generates two output files:

1. **JSON Results**: Contains detailed information including:
   - Full prompts and completions
   - Token usage statistics
   - Metadata for each completion
   - Error information for failed items
   - Timestamp and run information

2. **CSV Export**: Contains a simplified view for easy analysis:
   - Prompt IDs and prompt text
   - Completions (2 per prompt)
   - Token usage
   - Estimated cost
   - Status and error messages

## Python API Usage

For more advanced usage, you can directly use the `BatchProcessor` class in your Python code:

```python
from utils.batch_processor import BatchProcessor

# Create prompts
prompts = [
    {"prompt_id": "test1", "prompt": "What is RLHF?"},
    {"prompt_id": "test2", "prompt": "Explain constitutional AI"}
]

# Initialize processor
processor = BatchProcessor(
    output_dir="data/custom_batch",
    max_workers=2,
    temperature=0.8
)

# Process batch
results = processor.process_batch(prompts)

# Access results
print(f"Success rate: {results['summary']['success_rate']}%")
print(f"Total tokens: {results['summary']['total_tokens']}")
```

## Examples

See `prompts/sample_batch.json` for an example prompt file you can use as a template.

## Troubleshooting

1. **API Key Issues**: Make sure your DeepSeek API key is set in your environment
2. **Rate Limits**: If hitting rate limits, reduce the number of max workers
3. **Memory Issues**: For very large batches, process in smaller chunks 