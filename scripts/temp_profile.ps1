# Basic PowerShell Profile with minimal configuration

# History settings
$global:MaximumHistoryCount = 2000

# Add any custom functions or aliases below this line
# ===================================================

# Example function for directory navigation
function cdp { Set-Location D:\Projects }

# Example function for quick git commands
function gs { git status }
function gaa { git add --all }
function gcm { param([Parameter(Mandatory)]$message) git commit -m $message }
function gp { git push }
function gl { git pull }

# Display welcome message
Write-Host "PowerShell profile loaded successfully." -ForegroundColor Green 