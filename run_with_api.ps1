#!/usr/bin/env pwsh
# PowerShell script to run the RLHF dashboard with the DeepSeek API key

# Set the API key as an environment variable
# $env:DEEPSEEK_API_KEY = "your-api-key-here"  # Set your API key
$env:MODEL_ID = "deepseek-chat"

Write-Host "Starting RLHF Dashboard with DeepSeek API integration..." -ForegroundColor Green
Write-Host "API Key: $env:DEEPSEEK_API_KEY (Using DeepSeek API)" -ForegroundColor Green
Write-Host "Model: $env:MODEL_ID" -ForegroundColor Green

# Run the dashboard
python run_dashboard.py

# Clean up environment variables when done
Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:\MODEL_ID -ErrorAction SilentlyContinue 