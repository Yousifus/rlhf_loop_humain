#!/usr/bin/env pwsh
# PowerShell script to run the full RLHF pipeline

# Set the API key as an environment variable
# $env:DEEPSEEK_API_KEY = "your-api-key-here"  # Set your API key
$env:MODEL_ID = "deepseek-chat"

# Display banner
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "       RLHF TRAINING PIPELINE                         " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Process annotations into training data
Write-Host "Step 1: Processing annotations into training data..." -ForegroundColor Yellow
$dataProcessOutput = python utils/data_processor.py 2>&1
Write-Host $dataProcessOutput

# Step 2: Train the reward model
Write-Host "Step 2: Training reward model from annotations..." -ForegroundColor Yellow
$trainingOutput = python scripts/train_reward_model.py --dry-run 2>&1
Write-Host $trainingOutput

# Step 3: Display statistics
Write-Host "Step 3: Analyzing RLHF dataset..." -ForegroundColor Yellow

# Create models directory if it doesn't exist
if (-not (Test-Path -Path "models")) {
    New-Item -Path "models" -ItemType Directory | Out-Null
}

# Look for training history file
$historyFile = "models/reward_model/training_history.json"
if (Test-Path -Path $historyFile) {
    $trainingHistory = Get-Content -Path $historyFile | ConvertFrom-Json
    Write-Host "Training history summary:" -ForegroundColor Green
    Write-Host "- Epochs completed: $($trainingHistory.epochs)" -ForegroundColor Green
    Write-Host "- Final training accuracy: $($trainingHistory.train_accuracy[-1])" -ForegroundColor Green
    Write-Host "- Final validation accuracy: $($trainingHistory.val_accuracy[-1])" -ForegroundColor Green
    Write-Host "- Examples used: $($trainingHistory.examples_seen)" -ForegroundColor Green
} else {
    Write-Host "No training history found. Please run the training step first." -ForegroundColor Red
}

# Step 4: Offer to run the dashboard
Write-Host ""
Write-Host "Step 4: Would you like to run the dashboard to see the results?" -ForegroundColor Yellow
$runDashboard = Read-Host "Run dashboard? (y/n)"

if ($runDashboard -eq "y" -or $runDashboard -eq "Y") {
    Write-Host "Starting RLHF Dashboard with DeepSeek API integration..." -ForegroundColor Green
    streamlit run scripts/run_dashboard.py
}

# Clean up environment variables when done
Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
Remove-Item Env:\MODEL_ID -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "RLHF Pipeline completed." -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan 