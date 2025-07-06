# Simple script to start the RLHF dashboard with DeepSeek API enabled
Write-Host "Starting RLHF Dashboard with DeepSeek API..." -ForegroundColor Cyan

# Set the DeepSeek API key
# $env:DEEPSEEK_API_KEY = "your-api-key-here"  # Set your API key

# Stop any running Streamlit processes
try {
    Get-Process -Name "streamlit" -ErrorAction SilentlyContinue | Stop-Process -ErrorAction SilentlyContinue
    Write-Host "Stopped any running Streamlit processes." -ForegroundColor Yellow
} catch {
    # No streamlit processes running, that's fine
}

# Run the dashboard
python run_dashboard.py 