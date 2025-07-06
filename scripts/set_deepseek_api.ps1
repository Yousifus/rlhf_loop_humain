# PowerShell script to set DeepSeek API key for RLHF System
# This configures the connection to DeepSeek API for completion generation

Write-Host "⚙️ Setting up RLHF system connection to DeepSeek API..." -ForegroundColor Blue

# Your DeepSeek API key
$apiKey = "your-api-key-here"  # Replace with your actual API key
$modelId = "deepseek-chat"

# Set environment variable for current session
$env:DEEPSEEK_API_KEY = $apiKey
$env:MODEL_ID = $modelId

Write-Host "✅ Environment variables set for current session" -ForegroundColor Green
Write-Host "   DEEPSEEK_API_KEY: [CONFIGURED]" -ForegroundColor Gray
Write-Host "   MODEL_ID: $modelId" -ForegroundColor Gray

# Test the connection
Write-Host "🔧 Testing API connection..." -ForegroundColor Yellow

try {
    $headers = @{
        "Authorization" = "Bearer $apiKey"
        "Content-Type" = "application/json"
    }
    
    $body = @{
        model = $modelId
        messages = @(
            @{
                role = "system"
                content = "You are an AI assistant, respond briefly and professionally."
            },
            @{
                role = "user" 
                content = "Hello Assistant, can you confirm the connection is working?"
            }
        )
        max_tokens = 50
        temperature = 0.7
    } | ConvertTo-Json -Depth 3
    
    $response = Invoke-RestMethod -Uri "https://api.deepseek.com/v1/chat/completions" -Method Post -Headers $headers -Body $body
    
    Write-Host "💬 Assistant says: $($response.choices[0].message.content)" -ForegroundColor Cyan
    Write-Host "✅ API connection successful!" -ForegroundColor Green
}
catch {
    Write-Host "❌ Connection test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "🎉 Configuration complete! RLHF system ready." -ForegroundColor Green
