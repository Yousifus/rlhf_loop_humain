# Setting Up DeepSeek API for the RLHF Dashboard

This guide will help you obtain a DeepSeek API key and set it up for use with the RLHF Pipeline Monitor.

## Getting a DeepSeek API Key

1. Visit [DeepSeek](https://platform.deepseek.com/) and create an account or log in
2. Navigate to your account settings or API section
3. Generate a new API key
4. Copy the API key to your clipboard

## Using Your DeepSeek API Key with the Dashboard

### Method 1: Using the PowerShell Script (Recommended)

1. Open PowerShell in your project directory
2. Run the following command, replacing `your_api_key_here` with your actual DeepSeek API key:

```powershell
.\set_deepseek_key.ps1 -ApiKey "your_api_key_here"
```

This will:
- Set the environment variable for your current PowerShell session
- Stop any running Streamlit processes
- Start the dashboard with the API key active

### Method 2: Setting Environment Variable Manually

#### Temporary (Current PowerShell Session Only)

```powershell
$env:DEEPSEEK_API_KEY = "your_api_key_here"
python run_dashboard.py
```

#### Permanent (System-Wide)

1. Search for "Environment Variables" in Windows search
2. Click on "Edit the system environment variables"
3. Click on "Environment Variables"
4. Under "User variables", click "New"
5. Variable name: `DEEPSEEK_API_KEY`
6. Variable value: `your_api_key_here`
7. Click OK on all dialogs

After setting the permanent environment variable, restart any open terminals and the dashboard.

## Verifying API Connection

Once the dashboard is running with your API key:

1. Navigate to the "Chat Interface" tab
2. Try asking a question in the chat input
3. If the API connection is working, you'll see a real-time streaming response
4. If there's an issue, you'll see the fallback message system

## Troubleshooting

- If you're seeing "DeepSeek API key not found" messages, ensure you've correctly set the environment variable
- Check the PowerShell console for any error messages during API requests
- Make sure your DeepSeek API key is valid and has not expired
- Verify your internet connection as the API requires web access 