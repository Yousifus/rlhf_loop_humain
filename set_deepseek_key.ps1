# PowerShell script to set DeepSeek API key and launch the dashboard

# Hardcoded DeepSeek API key
$ApiKey = "your-api-key-here"  # Replace with your actual API key

Write-Host "Setting DeepSeek API key..."
$env:DEEPSEEK_API_KEY = $ApiKey

# Verify the key was set
if ($env:DEEPSEEK_API_KEY -eq $ApiKey) {
    Write-Host "DeepSeek API key set successfully!" -ForegroundColor Green
} else {
    Write-Host "Failed to set DeepSeek API key." -ForegroundColor Red
    exit 1
}

# Stop any running Streamlit processes
try {
    Get-Process -Name "streamlit" -ErrorAction SilentlyContinue | Stop-Process -ErrorAction SilentlyContinue
    Write-Host "Stopped any running Streamlit processes." -ForegroundColor Yellow
} catch {
    # No streamlit processes running, that's fine
}

# Launch the dashboard
Write-Host "Starting RLHF Attunement Dashboard with DeepSeek API enabled..."
python run_dashboard.py 