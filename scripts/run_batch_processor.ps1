# PowerShell script to run the batch processor for RLHF prompts

param (
    [string]$InputFile,
    [int]$MaxWorkers = 3,
    [float]$Temperature = 0.7,
    [float]$TopP = 0.9,
    [int]$MaxTokens = 256,
    [string]$OutputDir = "data/batch_results"
)

# Check if DeepSeek API key is set
if (-not $env:DEEPSEEK_API_KEY) {
    Write-Host "DeepSeek API key not found in environment variables" -ForegroundColor Yellow
    
    # Prompt user for action
    $setup = Read-Host "Would you like to set up the DeepSeek API key now? (y/n)"
    
    if ($setup -eq "y") {
        # Run the setup script
        python utils/setup_deepseek.py
        
        # Source the environment file if it exists
        $envFile = "$env:USERPROFILE\.deepseek_env"
        if (Test-Path $envFile) {
            Get-Content $envFile | ForEach-Object {
                if (-not $_.StartsWith('#') -and $_.Contains('=')) {
                    $name, $value = $_.Split('=', 2)
                    Set-Item -Path "env:$name" -Value $value
                }
            }
            Write-Host "DeepSeek API key loaded from $envFile" -ForegroundColor Green
        } else {
            Write-Host "No .deepseek_env file found. Please set the DEEPSEEK_API_KEY manually." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Running without DeepSeek API integration. API calls will likely fail." -ForegroundColor Yellow
    }
}

if (-not $InputFile) {
    Write-Host "Error: Input file is required." -ForegroundColor Red
    Write-Host "Usage: .\run_batch_processor.ps1 -InputFile path/to/prompts.json [-MaxWorkers 3] [-Temperature 0.7] [-TopP 0.9] [-MaxTokens 256] [-OutputDir 'data/batch_results']" -ForegroundColor Cyan
    exit 1
}

if (-not (Test-Path $InputFile)) {
    Write-Host "Error: Input file not found: $InputFile" -ForegroundColor Red
    exit 1
}

# Create a Python script to process the batch
$batchScript = @"
import json
import sys
from utils.batch_processor import BatchProcessor

input_file = '$InputFile'
max_workers = $MaxWorkers
temperature = $Temperature
top_p = $TopP
max_tokens = $MaxTokens
output_dir = '$OutputDir'

# Load prompts from file
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Validate input format
if isinstance(data, list):
    prompts = data
    if len(prompts) > 0 and not isinstance(prompts[0], dict):
        # Convert simple strings to dicts with prompt key
        prompts = [{'prompt': p} if isinstance(p, str) else p for p in prompts]
elif isinstance(data, dict) and 'prompts' in data:
    prompts = data['prompts']
else:
    print("Error: Input file must contain a list of prompts or a dict with a 'prompts' key")
    sys.exit(1)

# Initialize and run batch processor
processor = BatchProcessor(
    output_dir=output_dir,
    max_workers=max_workers,
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens
)

print(f"Starting batch processing of {len(prompts)} prompts...")
batch_results = processor.process_batch(prompts)

# Print summary
print("\nBatch processing complete!")
print(f"Success rate: {batch_results['summary']['success_rate']}%")
print(f"Total tokens: {batch_results['summary']['total_tokens']}")
print(f"Estimated cost: ${batch_results['summary']['estimated_cost']:.6f}")
print(f"Results saved to: {batch_results['summary']['results_file']}")
print(f"CSV export: {batch_results['summary']['csv_file']}")
"@

# Write the script to a temporary file
$tempScript = New-TemporaryFile
Set-Content -Path $tempScript.FullName -Value $batchScript

# Run the script
Write-Host "Starting batch processing with the following settings:" -ForegroundColor Cyan
Write-Host "  Input File: $InputFile" -ForegroundColor Cyan
Write-Host "  Max Workers: $MaxWorkers" -ForegroundColor Cyan
Write-Host "  Temperature: $Temperature" -ForegroundColor Cyan
Write-Host "  Top-P: $TopP" -ForegroundColor Cyan
Write-Host "  Max Tokens: $MaxTokens" -ForegroundColor Cyan
Write-Host "  Output Directory: $OutputDir" -ForegroundColor Cyan
Write-Host ""

python $tempScript.FullName

# Clean up temporary file
Remove-Item $tempScript.FullName -Force 