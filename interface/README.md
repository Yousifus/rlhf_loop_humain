# 🛠️ RLHF Pipeline Monitor - Streamlit Dashboard

**Comprehensive analytics dashboard for deep RLHF analysis and admin tasks**

## 🎯 **Quick Start - Get Streamlit Running**

### 🚀 **Super Simple Setup**
```bash
# 1. Enable rich demo data
python scripts/demo_mode.py enable

# 2. Launch Streamlit dashboard
streamlit run scripts/run_dashboard.py

# 🎉 Open: http://localhost:8501
```

### 📚 **Want both React + Streamlit?**
👉 **[See Main QUICK_START.md](../QUICK_START.md)** for complete dual-interface setup!

---

## 📊 **Why Choose Streamlit Dashboard?**

Perfect for:
- **📊 Deep analysis** - Comprehensive calibration & drift analysis
- **🎮 Portfolio demos** - Rich 450+ prompts with 6-month evolution
- **🔧 Admin tasks** - Batch processing, model training, system management
- **📈 Research work** - Advanced visualizations and detailed insights
- **🎯 Feature development** - Rapid prototyping of new analytics

### 🔥 **Streamlit-Only Features**
- **🎮 Rich Demo Mode** - 450+ prompts across 6 domains with realistic evolution
- **🔧 Advanced Admin Tools** - Batch processing, model training workflows
- **📊 Deep Analytics** - Comprehensive calibration and drift analysis
- **📈 Research Features** - Detailed error analysis and categorization

---

## 🌟 **Streamlit vs React**

| Feature | 🛠️ Streamlit | 🚀 React | Best For |
|---------|-------------|----------|----------|
| **Demo Data** | **450+ rich prompts** | Real data only | **Portfolio showcases** |
| **Analytics Depth** | **Comprehensive** | Essential metrics | **Research & analysis** |
| **Admin Tools** | **Full featured** | Basic settings | **System management** |
| **Mobile Experience** | Desktop-optimized | **Perfect mobile** | **Daily monitoring** |
| **Load Speed** | 3-5 seconds | **Sub-second** | **Quick checks** |
| **AI Chat** | Not available | **Built-in** | **Model testing** |

**🎯 Recommendation:** Use both! React for daily work, Streamlit for deep analysis.

---

## 🎮 **Rich Demo Mode** *(Streamlit Exclusive)*

### ✨ **Portfolio Showcase Features**
- **📊 450+ Diverse Prompts** across 6 professional domains
- **📈 Realistic Model Evolution** from 58% → 87% accuracy over 6 months
- **🎯 Advanced Calibration** showing confidence alignment improvements
- **🌟 Multi-Domain Analysis** - Programming, AI/ML, Ethics, Creative, Career, Tech
- **📅 Temporal Patterns** - Authentic usage trends and learning progression

### 🎨 **Content Categories**
| Domain | Prompts | Examples |
|--------|---------|----------|
| 🐍 **Programming** | 22% | Python algorithms, debugging, architecture |
| 🤖 **AI/ML Concepts** | 29% | Neural networks, ethics, safety explanations |
| 💭 **Ethics & Philosophy** | 24% | AI regulation, privacy, complex reasoning |
| ✨ **Creative Writing** | 12% | Stories, poetry, worldbuilding scenarios |
| 💼 **Career Development** | 7% | Professional advice, interviews, growth |
| 🌐 **Current Events** | 6% | Climate tech, quantum computing, future trends |

---

## 🏗️ **System Architecture**

The RLHF Pipeline Monitor provides comprehensive visualization across the complete RLHF lifecycle:

### 🏭 **4-Phase Pipeline Structure**

#### **📊 Phase 1: Data Collection**
- **Rich Data Ingestion** - 450+ annotated preference pairs
- **Quality Assessment** - Domain-specific metrics and validation  
- **Annotation Analytics** - Human feedback pattern analysis

#### **🚀 Phase 2: Training**
- **Model Evolution Tracking** - 4 checkpoint progression
- **Performance Monitoring** - Real-time accuracy and loss analysis
- **Resource Management** - Training efficiency optimization

#### **🎯 Phase 3: Evaluation**
- **Advanced Calibration** - Reliability diagrams and ECE analysis
- **Drift Detection** - Content and performance shift monitoring
- **Multi-Domain Assessment** - Category-specific performance tracking

#### **🌐 Phase 4: Deployment**
- **System Health Dashboard** - Operational status monitoring
- **Performance Analytics** - Production metrics and insights
- **Quality Assurance** - Continuous validation and alerts

---

## 📁 **Project Structure**

```
interface/
├── scripts/
│   └── run_dashboard.py          # 🎯 Main Streamlit entry point
├── components/                   # 🔧 Shared utilities
│   ├── data_loader.py           # Data loading and preprocessing  
│   ├── utils.py                 # Helper functions and constants
│   └── visualization.py         # Reusable visualization components
├── sections/                    # 📊 Dashboard pages
│   ├── overview.py              # System health and metrics overview
│   ├── annotation.py            # Annotation quality and management
│   ├── model_insights.py        # Dataset analytics and insights
│   ├── model_evolution.py       # Training progress and evolution
│   ├── calibration.py           # Calibration analysis and validation
│   ├── drift_analysis.py        # Drift detection and monitoring
│   ├── alignment.py             # Model alignment and safety metrics
│   └── chat.py                  # Debug chat interface
└── dashboard_README.md          # Detailed feature documentation
```

---

## 🚀 **Running the Dashboard**

### 🎮 **Demo Mode** *(Recommended for showcases)*
```bash
# Enable rich demo data
python scripts/demo_mode.py enable

# Launch dashboard
streamlit run scripts/run_dashboard.py

# 🎉 Access: http://localhost:8501
```

### 🛠️ **Development Mode**
```bash
# Configure for real development
python scripts/demo_mode.py disable
export DEEPSEEK_API_KEY="your_key_here"  # Optional
streamlit run scripts/run_dashboard.py
```

### 🔍 **Debug Mode**
```bash
# Access hidden debug features
# http://localhost:8501?debug=chat
```

---

## 🎯 **Key Features**

### 📊 **Comprehensive Analytics**
- **Performance Tracking** - Multi-metric model evaluation
- **Calibration Analysis** - Confidence alignment with reliability diagrams
- **Drift Detection** - Statistical monitoring of model degradation
- **Domain Analysis** - Category-specific performance insights

### 🎨 **HUMAIN OS Design**
- **Professional Styling** - HUMAIN teal (`#1DB584`) color scheme
- **Clean Interface** - Professional layouts with intuitive navigation
- **Rich Visualizations** - Interactive Plotly charts and graphs
- **Responsive Design** - Desktop-optimized with mobile compatibility

### 🔧 **Advanced Tools**
- **Batch Processing** - Large-scale data processing workflows
- **Model Training** - Training pipeline monitoring and management
- **API Integration** - DeepSeek/OpenAI API key management
- **Data Export** - Comprehensive data export capabilities

---

## 🔄 **Dual Interface Benefits**

### 🛠️ **Use Streamlit For:**
- **📊 Deep research** - Comprehensive analytics and insights
- **🎮 Portfolio demos** - Rich demo data with 6-month evolution
- **🔧 System administration** - Batch processing and model training
- **📈 Feature development** - Rapid prototyping of new analytics
- **🎯 Educational purposes** - Learning RLHF concepts and workflows

### 🚀 **Use React For:**
- **👨‍💼 Daily monitoring** - Quick insights and real-time updates
- **📱 Mobile access** - Check metrics on-the-go
- **💬 AI interactions** - Built-in chat interface for model testing
- **⚙️ Quick configuration** - Visual API setup without environment variables
- **🎤 Live demos** - Fast, professional presentations

**🌟 Run both simultaneously** for the complete RLHF experience!

---

## 🛠️ **Technical Requirements**

### 📦 **Dependencies**
```bash
pip install -r requirements.txt
```

Key packages:
- **streamlit** - Web application framework
- **plotly** - Interactive visualizations  
- **pandas** - Data processing and analysis
- **transformers** - AI model integration
- **scikit-learn** - Machine learning utilities

### ⚙️ **Configuration**
- **Environment Variables** - Optional API key configuration
- **Demo Mode** - Rich showcase data with `python scripts/demo_mode.py enable`
- **Debug Mode** - Development features via URL parameters
- **Custom Styling** - HUMAIN OS professional theme

---

## 📚 **Learn More**

- **📖 [Main README](../README.md)** - Complete project overview
- **⚡ [QUICK_START.md](../QUICK_START.md)** - 60-second setup for both interfaces
- **🚀 [web_modern/README.md](../web_modern/README.md)** - React dashboard guide
- **📊 [dashboard_README.md](dashboard_README.md)** - Detailed Streamlit features

---

## 🤝 **Contributing**

```bash
# Setup Streamlit development
pip install -r requirements.txt
python scripts/demo_mode.py enable

# Run development server
streamlit run scripts/run_dashboard.py

# Test with rich demo data
# Access: http://localhost:8501
```

---

## 🎉 **Ready for Deep Analytics?**

```bash
# Enable rich demo data
python scripts/demo_mode.py enable

# Launch comprehensive dashboard
streamlit run scripts/run_dashboard.py

# 🛠️ Open: http://localhost:8501
```

**Welcome to the most comprehensive RLHF analytics platform! 📊🔧🎮**

---

*Built with ❤️ using Streamlit, Plotly, and the HUMAIN design system* 