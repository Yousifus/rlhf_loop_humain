# 🎛️ RLHF Pipeline Monitor
### *Professional Multi-Provider AI Dashboard*

The RLHF Pipeline Monitor provides comprehensive monitoring and analysis across the complete RLHF lifecycle, from data collection through deployment. **Now featuring multi-provider AI integration with LM Studio, DeepSeek, and OpenAI support!**

---

## 🔥 **NEW: Multi-Provider AI Integration**

### 🏠 **LM Studio - Privacy-First Local AI**
- **Zero Configuration** - Auto-detects LM Studio servers and available models
- **Complete Privacy** - All processing stays on your machine
- **7+ Models Supported** - Llama, Qwen, Mistral, and more
- **No Internet Required** - Perfect for offline demos and secure environments

### 🌐 **Cloud Provider Support**
- **DeepSeek API** - Cost-effective, powerful language models
- **OpenAI API** - Industry-standard GPT models
- **In-Dashboard Configuration** - Secure API key management with real-time validation

---

## 🎯 **Core Features**

### 📊 **4-Phase Pipeline Dashboard**
The dashboard provides comprehensive monitoring across the complete RLHF lifecycle:

#### **📥 Data Collection Phase**
- **📋 Annotation Interface** - Interactive human feedback collection
- **📊 Dataset Statistics** - Comprehensive data quality metrics
- **📈 Data Quality Metrics** - Real-time validation and monitoring

#### **🔧 Training Phase**
- **🚀 Training Status** - Live model training progress tracking
- **📈 Loss Curves** - Advanced training metrics and optimization
- **💭 Model Insights** - Deep performance analysis and evolution

#### **📋 Evaluation Phase** 
- **🎯 Model Performance** - Accuracy tracking and alignment analysis
- **📊 Calibration Analysis** - Confidence estimation and reliability diagrams
- **🌊 Drift Detection** - Content and performance shift monitoring

#### **🚀 Deployment Phase**
- **⚡ System Overview** - Operational status and health monitoring
- **📊 Production Metrics** - Real-time performance analytics
- **🔍 System Health** - Infrastructure monitoring and alerts

### 🤖 **AI Provider Management**
- **🔄 Real-Time Detection** - Auto-discovery of available providers
- **🧪 One-Click Testing** - Instant provider validation with diagnostics
- **🔍 Debug Tools** - Professional troubleshooting and connection monitoring
- **🔄 Seamless Switching** - Change providers without restart

---

## 🚀 **Getting Started**

### 🏠 **Local AI Setup (LM Studio)**
```bash
# 1. Download and install LM Studio from lmstudio.ai
# 2. Browse lmstudio.ai/models and download any model (e.g., Llama 3.2, Mistral, Qwen)
# 3. Go to Developer tab → Start Server (port 1234)
# 4. Launch dashboard - auto-detected instantly!
streamlit run scripts/run_dashboard.py
```

### 🌐 **Cloud AI Setup**
```bash
# 1. Get API key from DeepSeek or OpenAI
# 2. Launch dashboard
streamlit run scripts/run_dashboard.py
# 3. Expand "🔑 API Key Configuration" in sidebar
# 4. Enter your API key and test connection
```

### 🎮 **Rich Demo Mode**
   ```bash
# Enable comprehensive demo with 450+ prompts
python scripts/demo_mode.py enable
streamlit run scripts/run_dashboard.py

# Access at: http://localhost:8501
# Debug mode: http://localhost:8501?debug=chat
```

---

## 🛠️ **Professional Features**

### 🔑 **Smart API Management**
- **Secure Configuration** - Password-protected API key inputs
- **Environment Sync** - Automatic environment variable management
- **Real-Time Validation** - Live connection testing and diagnostics
- **Provider Status** - Visual indicators for availability and health

### 📊 **Advanced Analytics**
- **Interactive Visualizations** - Professional charts with Plotly integration
- **Performance Metrics** - Real-time accuracy, calibration, and drift analysis  
- **Multi-Domain Intelligence** - Category-specific performance tracking
- **Temporal Analysis** - Model evolution and improvement trends

### 🎨 **Professional UI/UX**
- **HUMAIN Branding** - Clean, modern interface with professional styling
- **Responsive Design** - Mobile-friendly layout with accessibility features
- **Performance Optimized** - Fast loading and smooth interactions
- **Error Handling** - Comprehensive validation and user-friendly error messages

---

## 🎯 **Use Cases**

### 💼 **Portfolio Demonstrations**
- **🎬 Instant Showcases** - Rich demo mode with 6 months of realistic data
- **🏠 Offline Presentations** - LM Studio for complete privacy and zero dependencies
- **📊 Professional Interface** - Production-ready dashboard perfect for interviews

### 🔬 **Research & Development**
- **🧠 RLHF Methodology** - Complete implementation of human feedback learning
- **📈 Model Calibration** - Advanced confidence estimation and reliability analysis
- **🌊 Drift Monitoring** - Systematic tracking of model degradation patterns

### 🏢 **Enterprise Applications**
- **⚡ Production Monitoring** - Real-time model performance tracking
- **🛡️ Security Options** - Local processing with LM Studio or secure cloud APIs
- **📊 Business Analytics** - Content performance across domains and categories

---

## 📚 **Data Sources & Integration**

### 🎮 **Demo Mode Data**
- **450+ Prompts** - Comprehensive dataset across 6 professional domains
- **Realistic Evolution** - 6 months of authentic model improvement (58% → 87%)
- **Rich Metadata** - Complete annotation and reflection data

### 📊 **Production Data**
- **Vote Logs** - Human preference annotations (`data/vote_logs/`)
- **Predictions** - Model output and confidence data (`data/predictions.jsonl`)
- **Reflections** - System introspection and analysis (`data/reflection_data.jsonl`)
- **Calibration** - Model confidence tracking (`models/calibration_log.json`)

---

## 🔧 **Troubleshooting**

### 🏠 **LM Studio Issues**
If LM Studio is not detected:
1. **Start LM Studio** - Ensure the desktop app is running
2. **Download a Model** - Visit [lmstudio.ai/models](https://lmstudio.ai/models) and download any model
3. **Load the Model** - Load your downloaded model in LM Studio
4. **Enable API Server** - Go to Developer tab → Start Server
5. **Check Port** - Ensure server is running on port 1234 (default)
6. **Refresh Providers** - Click "🔄 Refresh Providers" in dashboard

### 🌐 **Cloud API Issues**
If cloud providers show as unavailable:
1. **Verify API Key** - Check key validity on provider platform
2. **Check Network** - Ensure internet connectivity
3. **Use Debug Tools** - Expand "🔍 Debug Info" for detailed diagnosis
4. **Test Connection** - Use "🧪 Test Provider" button for validation

### 📊 **Data Issues**
If dashboard shows no data:
1. **Enable Demo Mode** - Run `python scripts/demo_mode.py enable`
2. **Check File Permissions** - Ensure read access to data files
3. **Validate JSON** - Check for corrupted data files
4. **Review Logs** - Check console output for specific errors

---

## 🎛️ **Dashboard Navigation**

### 🌐 **Main Interface**
- **Provider Selection** - Choose between LM Studio, DeepSeek, or OpenAI
- **API Configuration** - Secure key management and provider setup
- **Pipeline Phases** - Navigate between Data Collection, Training, Evaluation, Deployment

### 🔍 **Debug Mode**
Access enhanced debugging tools:
- **URL**: `http://localhost:8501?debug=chat`
- **Features**: Direct chat interface with provider testing
- **Purpose**: Development and troubleshooting

---

## 📈 **Performance & Optimization**

### ⚡ **Optimized Loading**
- **Smart Caching** - Intelligent data loading and provider management
- **Lazy Loading** - Performance optimization for large datasets
- **Memory Efficiency** - Optimized resource usage and garbage collection

### 📊 **Scalable Architecture**
- **Modular Design** - Component-based architecture for maintainability
- **Provider Abstraction** - Universal interface across all AI providers
- **Future-Ready** - Designed for easy addition of new providers and features

---

## 🤝 **Contributing**

### 🛠️ **Development Setup**
```bash
# Clone repository
git clone https://github.com/Yousifus/rlhf_loop_humain.git
cd rlhf_loop_humain

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run scripts/run_dashboard.py
```

### 📝 **Code Standards**
- **Professional Styling** - Follow HUMAIN design system guidelines
- **Error Handling** - Comprehensive validation and user feedback
- **Documentation** - Clear docstrings and inline comments
- **Testing** - Comprehensive test coverage for new features

---

## 📞 **Support**

For technical support and feature requests:
- **📧 Email**: [yoawlaki@gmail.com](mailto:yoawlaki@gmail.com)
- **🐙 GitHub**: [github.com/Yousifus/rlhf_loop_humain](https://github.com/Yousifus/rlhf_loop_humain)
- **📋 Issues**: Use GitHub Issues for bug reports and feature requests

---

<div align="center">

### 🌟 **Professional RLHF Dashboard** 🌟
*Multi-Provider AI Integration • Privacy-First Local Processing • Enterprise-Ready*

**Perfect for portfolio showcases, technical interviews, and production deployment**

</div> 