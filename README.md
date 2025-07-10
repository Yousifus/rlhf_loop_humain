# RLHF Loop System

A learning project that implements a basic RLHF (Reinforcement Learning from Human Feedback) pipeline with annotation collection, model monitoring, and data analysis tools.

## What This Is

This project provides:
- **Annotation System**: Collect human feedback on AI model responses with detailed quality ratings
- **Multiple AI Providers**: Connect to DeepSeek, OpenAI, LM Studio, or Grok (X.AI)
- **Two Interfaces**: A React web dashboard and a Streamlit analytics interface
- **Basic Analytics**: Track model performance, calibration, and data drift
- **SQLite Storage**: Store annotations and feedback data locally

## What This Is Not

- Not a production system
- Not "enterprise-ready" or "professional-grade"
- Not the only or best RLHF tool available
- Not extensively tested or optimized for scale

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/Yousifus/rlhf_loop_humain.git
cd rlhf_loop_humain
pip install -r requirements.txt

# Start the backend API
cd api_backend
python main.py

# In another terminal, start the React interface
cd web_modern
npm install
npm run dev

# Access at http://localhost:3000
```

## AI Provider Setup

### LM Studio (Local)
1. Download LM Studio from lmstudio.ai
2. Load any model and start the server
3. The system will auto-detect it

### Cloud Providers
Add API keys through the Settings page:
- **DeepSeek**: Get key from platform.deepseek.com
- **OpenAI**: Get key from platform.openai.com
- **Grok**: Get key from console.x.ai

## Current Features

### Annotation System
- Rate response quality on multiple dimensions (0-10 scale)
- Select reasons for preferring one response over another
- Indicate confidence in your choices
- Store detailed feedback in SQLite database

### Model Integration
- Test different AI providers on the same prompts
- Compare responses side by side
- Switch between models easily

### Basic Analytics
- View annotation history
- Simple performance tracking
- Basic calibration analysis
- Data drift detection

### Demo Data
- Includes sample prompts and responses for testing
- Enable with: `python scripts/demo_mode.py enable`

## File Structure

```
├── api_backend/          # FastAPI backend server
├── web_modern/           # React frontend
├── interface/            # Streamlit interface
├── utils/                # Shared utilities
├── data/                 # SQLite database and demo data
└── scripts/              # Helper scripts
```

## Technology Used

- **Backend**: FastAPI, SQLite
- **Frontend**: React, Next.js, TypeScript, Tailwind CSS
- **Analytics**: Streamlit, Plotly
- **AI Integration**: Direct API calls to various providers

## Current Limitations

- Only tested with small datasets
- Limited error handling
- Basic UI/UX design
- No user authentication
- Local storage only
- Manual deployment required

## Development Status

This is a learning project built with AI assistance (Claude). It demonstrates basic RLHF concepts but is not intended for production use.

## Recent Updates

- Enhanced annotation interface with detailed feedback collection
- Added Grok (X.AI) provider support
- Improved SQLite data storage
- Added vote predictor training capabilities

## License

MIT License - see LICENSE file for details.

## Contributing

This is primarily a personal learning project. Feel free to fork and experiment, but no formal contribution process is established.