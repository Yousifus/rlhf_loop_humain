# PowerShell script to run the RLHF dashboard with DeepSeek API integration

# Initialize flags
$keySetInEnv = $false
$keySetInFile = $false

# Check if DeepSeek API key is already set in environment
if ($env:DEEPSEEK_API_KEY) {
    Write-Host "DeepSeek API key already found in environment variables" -ForegroundColor Green
    $keySetInEnv = $true
}

# Check if a .deepseek_env file exists
$envFile = "$env:USERPROFILE\.deepseek_env"
if (Test-Path $envFile) {
    $keySetInFile = $true
    Write-Host "Found DeepSeek API key file at $envFile" -ForegroundColor Cyan
    
    # Load the key from file if not already in environment
    if (-not $keySetInEnv) {
        Get-Content $envFile | ForEach-Object {
            if (-not $_.StartsWith('#') -and $_.Contains('=')) {
                $name, $value = $_.Split('=', 2)
                Set-Item -Path "env:$name" -Value $value
            }
        }
        if ($env:DEEPSEEK_API_KEY) {
            Write-Host "DeepSeek API key loaded from $envFile" -ForegroundColor Green
            $keySetInEnv = $true
        }
    }
}

# Ask to set up if API key is still not available
if (-not $keySetInEnv) {
    Write-Host "DeepSeek API key not found in environment or config file" -ForegroundColor Yellow
    
    # Prompt user for action
    $setup = Read-Host "Would you like to set up the DeepSeek API key now? (y/n)"
    
    if ($setup -eq "y") {
        # Try up to 3 times to get a valid key
        $attempts = 0
        $maxAttempts = 3
        
        while ($attempts -lt $maxAttempts -and -not $keySetInEnv) {
            $attempts++
            
            # Get API key directly if setup script doesn't exist or on retry
            if (-not (Test-Path "utils/setup_deepseek.py") -or $attempts -gt 1) {
                Write-Host "`n=== DeepSeek API Setup ===" -ForegroundColor Cyan
                Write-Host "You need a DeepSeek API key to use the API features."
                Write-Host "You can get one by signing up at https://platform.deepseek.com"
                $apiKey = Read-Host "Enter your DeepSeek API key"
                
                if ($apiKey.Trim()) {
                    # Set the key in environment
                    $env:DEEPSEEK_API_KEY = $apiKey.Trim()
                    $keySetInEnv = $true
                    
                    # Save to config file
                    "DEEPSEEK_API_KEY=$apiKey" | Set-Content $envFile
                    Write-Host "DeepSeek API key saved to $envFile" -ForegroundColor Green
                } else {
                    Write-Host "API key cannot be empty. Try again..." -ForegroundColor Red
                }
            } else {
                # Run the setup script
                python utils/setup_deepseek.py
                
                # Check if the key was set by the script
                if (Test-Path $envFile) {
                    Get-Content $envFile | ForEach-Object {
                        if (-not $_.StartsWith('#') -and $_.Contains('=')) {
                            $name, $value = $_.Split('=', 2)
                            Set-Item -Path "env:$name" -Value $value
                        }
                    }
                    if ($env:DEEPSEEK_API_KEY) {
                        Write-Host "DeepSeek API key loaded from $envFile" -ForegroundColor Green
                        $keySetInEnv = $true
                    } else {
                        Write-Host "API key not set correctly. Let's try again." -ForegroundColor Yellow
                    }
                }
            }
        }
        
        if (-not $keySetInEnv) {
            Write-Host "Failed to set up DeepSeek API key after $maxAttempts attempts." -ForegroundColor Red
            Write-Host "The dashboard will run with simulation mode for completions." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Running without DeepSeek API integration. API features will use simulation." -ForegroundColor Yellow
    }
}

# Disable Streamlit telemetry
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

# Run the dashboard
Write-Host "`nStarting RLHF Attunement Dashboard..." -ForegroundColor Cyan
python run_dashboard.py 