# 🤖 RLHF Loop System
### *Reinforcement Learning from Human Feedback Platform*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.0+-blue.svg)](https://typescriptlang.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0+-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 **Project Overview**

A comprehensive **Reinforcement Learning from Human Feedback (RLHF)** system with predictive modeling, calibrated confidence scores, and real-time monitoring capabilities through a **RLHF Pipeline Monitor** dashboard.

### 🌟 **Key Features**

| Feature | Description | Status |
|---------|-------------|--------|
| 🎛️ **Pipeline Dashboard** | 4-phase RLHF pipeline monitoring interface | ✅ Complete |
| 🧠 **AI Vote Prediction** | Transformer-based preference modeling | ✅ Complete |
| 📊 **Drift Detection** | Model performance monitoring | ✅ Complete |
| ⚡ **Batch Processing** | Parallel prompt processing pipeline | ✅ Complete |
| 🎯 **Calibrated Confidence** | Temperature-scaled prediction confidence | ✅ Complete |
| 🔄 **Training Loop** | Continuous model improvement | ✅ Complete |

---

## 🏗️ **System Architecture**

```mermaid
graph TD
    subgraph "Data Collection Phase"
        A[📊 Data Ingestion] --> B[🔍 Quality Control]
        B --> C[📈 Dataset Analytics]
    end
    
    subgraph "Training Phase"
        D[🚀 Training Status] --> E[📊 Loss Monitoring]
        E --> F[⚙️ Resource Utilization]
    end
    
    subgraph "Evaluation Phase"
        G[🎯 Performance Metrics] --> H[📏 Calibration Analysis]
        H --> I[🔍 Drift Detection]
    end
    
    subgraph "Deployment Phase"
        J[🌐 Serving Status] --> K[📊 Production Metrics]
        K --> L[💻 System Health]
    end
    
    C --> D
    F --> G
    I --> J
    L --> A
    
    style A fill:#1DB584,color:#fff
    style D fill:#1DB584,color:#fff
    style G fill:#1DB584,color:#fff
    style J fill:#1DB584,color:#fff
```

---

## 🚀 **Quick Start**

### 📋 **Prerequisites**
```bash
Python 3.8+  |  Node.js 16+  |  Git
```

### ⚡ **Installation**
```bash
# Clone the repository
git clone https://github.com/Yousifus/rlhf_loop_humain.git
cd rlhf_loop_humain

# Install dependencies
pip install -r requirements.txt
npm install

# Launch the RLHF Pipeline Monitor
python scripts/run_dashboard.py
```

### 🎛️ **Dashboard Access**
```
🌐 Local:     http://localhost:8501
🔧 Interface: RLHF Pipeline Monitor (4-Phase Structure)
📊 Features:  Data Collection | Training | Evaluation | Deployment
🔍 Debug:     http://localhost:8501?debug=chat (Hidden chat interface)
```

---

## 📁 **Project Structure**

```
rlhf_loop_humain/
├── 🎨 web/              # Frontend components & interfaces
├── 🚀 scripts/          # Automation & deployment tools  
│   └── run_dashboard.py # Main RLHF Pipeline Monitor
├── 🧪 tests/            # Quality assurance & validation
├── ⚙️ config/           # Configuration management
├── 📚 docs/             # Technical documentation
├── 🤖 interface/        # Dashboard sections & components
│   ├── sections/        # 4-phase pipeline sections
│   └── components/      # Reusable UI components
├── 🧠 models/           # ML models & checkpoints
├── 📊 data/             # Training data & pipelines
├── 🛠️ utils/            # Core utilities & APIs
└── 📋 tasks/            # Task management system
```

---

## 🛠️ **Technology Stack**

### 🧠 **AI & Machine Learning**
- **🤗 Transformers** - BERT-based preference modeling
- **🔥 PyTorch** - Deep learning framework
- **📊 scikit-learn** - Model validation & metrics
- **📈 NumPy/Pandas** - Data processing & analysis

### 🎛️ **Dashboard & Visualization** 
- **🎨 Streamlit** - Web interface with HUMAIN styling
- **📊 Plotly** - Interactive data visualization
- **🎯 Matplotlib** - Statistical plotting
- **🎨 Custom CSS** - HUMAIN OS design system

### 🚀 **Infrastructure**
- **🐍 Python 3.8+** - Core runtime environment
- **📝 TypeScript** - Type-safe frontend development
- **⚡ PowerShell** - Windows automation scripts
- **🔧 Git** - Version control & collaboration

### 🔗 **AI API Integration**
- **🤖 DeepSeek** - Language model API
- **🧠 OpenAI** - GPT model integration
- **🏠 LM Studio** - Local model deployment
- **🔄 RESTful APIs** - Standardized integrations

---

## 📊 **RLHF Pipeline Monitor**

### 🏭 **4-Phase Dashboard Structure**
The dashboard is organized around the RLHF pipeline lifecycle:

#### **📊 Phase 1: Data Collection**
- **Data Ingestion Pipeline** - Real-time data streaming monitoring
- **Annotation Quality Control** - Inter-annotator agreement tracking
- **Dataset Analytics** - Data distribution analysis

#### **🚀 Phase 2: Training**
- **Training Status Dashboard** - Live training progress monitoring
- **Loss Curve Analysis** - Multi-metric performance tracking
- **Resource Utilization** - GPU/CPU/Memory monitoring

#### **🎯 Phase 3: Evaluation**
- **Performance Metrics** - Model accuracy and effectiveness
- **Calibration Analysis** - Confidence alignment validation
- **Drift Detection** - Statistical change monitoring

#### **🌐 Phase 4: Deployment**
- **Serving Status** - Production deployment monitoring
- **Production Metrics** - Live inference performance
- **System Health** - Infrastructure monitoring

### 🎨 **HUMAIN OS Design System**
- **Primary Color:** `#1DB584` (HUMAIN Teal)
- **Clean Interface:** Professional white backgrounds
- **Responsive Layout** - Mobile-friendly design
- **Performance Optimized** - Fast loading times

---

## 🎯 **Use Cases**

### 🏢 **Production Applications**
- **RLHF Pipeline Management** - Complete lifecycle monitoring
- **Model Performance Tracking** - Real-time accuracy monitoring
- **Quality Assurance** - Automated validation and testing
- **Deployment Monitoring** - Live system health tracking

### 🔬 **Research Applications**  
- **RLHF Methodology** - Human preference learning research
- **Model Calibration** - Confidence estimation studies
- **Drift Detection** - Model degradation analysis
- **Active Learning** - Efficient annotation strategies

---

## 📈 **Performance Metrics**

| Metric | Current Performance | Target |
|--------|-------------------|--------|
| 🎯 **Prediction Accuracy** | 85.3% | 90%+ |
| ⚡ **Response Time** | <200ms | <100ms |
| 📊 **Calibration Error** | 0.05 | <0.03 |
| 🔄 **Training Speed** | 2.5 hr/epoch | <2 hr/epoch |

---

## 🚀 **Getting Started**

### 🎛️ **RLHF Pipeline Monitor**
```bash
# Launch the main dashboard
python scripts/run_dashboard.py

# With API key setup (guided in-dashboard)
# 1. Open http://localhost:8501
# 2. Configure DeepSeek API key in sidebar
# 3. Access all 4 pipeline phases

# Debug mode with chat interface
http://localhost:8501?debug=chat
```

### 🧠 **Model Training**
```bash
# Prepare training data
python utils/vote_predictor/data_prep.py

# Train preference model
python scripts/train_reward_model.py

# Calibrate confidence scores
python utils/vote_predictor/calibrate.py
```

### 🔍 **Testing**
```bash
# Run complete test suite
python -m pytest tests/ -v

# Validate data connections
python tests/test_data_connections.py

# Check model performance
python tests/test_reflection_data.py
```

---

## 🛡️ **Features**

### 🔐 **Security & Privacy**
- **🔒 API Integration** - In-dashboard key management
- **📝 Audit Logging** - Activity tracking
- **🛡️ Data Protection** - Privacy-focused design

### 📊 **Monitoring & Analytics**
- **📈 Real-time Dashboards** - 4-phase pipeline monitoring
- **🚨 Alert Systems** - Issue detection
- **📋 Reporting Tools** - Performance insights
- **🔍 Diagnostic Tools** - System analysis

### ⚡ **Performance**
- **🚀 Optimized Processing** - Efficient batch operations
- **📊 Memory Management** - Resource optimization
- **⚡ Caching Systems** - Response time optimization

---

## 📚 **Documentation**

| Resource | Description | Link |
|----------|-------------|------|
| 🏗️ **Architecture Guide** | System design & components | [`docs/architecture.md`](docs/architecture.md) |
| 🚀 **Deployment Guide** | Setup instructions | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| 🧪 **Testing Guide** | QA procedures & validation | [`tests/README.md`](tests/README.md) |

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

### 🛠️ **Development Setup**
```bash
# Fork the repository
git clone https://github.com/yourusername/rlhf_loop_humain.git

# Create feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **🤗 Hugging Face** - Transformers library and model ecosystem
- **🎨 Streamlit** - Web application framework  
- **🔥 PyTorch** - Deep learning platform
- **🧠 OpenAI** - RLHF methodology research

---

## 📞 **Contact**

- **👨‍💻 Developer:** Yousifus
- **📧 Email:** [yoawlaki@gmail.com](mailto:yoawlaki@gmail.com)
- **🐙 GitHub:** [@Yousifus](https://github.com/Yousifus)
- **🔗 Repository:** [rlhf_loop_humain](https://github.com/Yousifus/rlhf_loop_humain)

---

<div align="center">

### 🌟 **Built with ❤️ for AI Research & Development** 🌟

*Transforming human feedback into intelligent systems*

[![GitHub Stars](https://img.shields.io/github/stars/Yousifus/rlhf_loop_humain?style=social)](https://github.com/Yousifus/rlhf_loop_humain/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Yousifus/rlhf_loop_humain?style=social)](https://github.com/Yousifus/rlhf_loop_humain/network/members)

</div>
