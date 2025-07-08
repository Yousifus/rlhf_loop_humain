# 🚀 **QUICK START** - *Get Running in 60 Seconds!*

## 🎯 **Choose Your Dashboard Experience**

### ⚡ **Option 1: Modern React Dashboard** *(Recommended)*
**Fast • Mobile-friendly • Built-in AI chat**

```bash
# 1. Basic setup
git clone https://github.com/Yousifus/rlhf_loop_humain.git
cd rlhf_loop_humain
pip install -r requirements.txt

# 2. Enable demo data
python scripts/demo_mode.py enable

# 3. Start backend (new terminal)
cd api_backend && python main.py

# 4. Start React (new terminal)  
cd web_modern && npm install && npm run dev

# 🎉 Open: http://localhost:3000
```

### 📊 **Option 2: Streamlit Dashboard** *(Feature-rich)*
**Comprehensive analytics • Rich demo data • Advanced tools**

```bash
# 1. Basic setup
git clone https://github.com/Yousifus/rlhf_loop_humain.git
cd rlhf_loop_humain
pip install -r requirements.txt

# 2. Enable demo data
python scripts/demo_mode.py enable

# 3. Launch dashboard
streamlit run scripts/run_dashboard.py

# 🎉 Open: http://localhost:8501
```

### 🌟 **Option 3: Both Dashboards** *(Ultimate setup)*
**Best of both worlds - use React for daily work, Streamlit for analysis**

```bash
# Follow Option 1 steps, then ALSO run:
streamlit run scripts/run_dashboard.py
# Now you have both! 🎉
```

---

## 🤖 **Add Your AI API Keys**

### 🚀 **React Dashboard** *(Visual setup)*
1. Go to **Settings** page in React dashboard
2. Add your **DeepSeek** or **OpenAI** API key
3. Click **Test** to verify connection
4. Use **Chat** interface immediately!

### 🏠 **LM Studio** *(Local AI - No API key needed)*
1. Download [LM Studio](https://lmstudio.ai)
2. Load any model (Mistral, Llama, etc.)
3. Start server (Developer tab → Start Server)
4. Both dashboards detect automatically!

---

## 🎮 **What You'll See**

- **📊 450+ Demo Prompts** across 6 professional domains
- **📈 Model Evolution** from 58% → 87% accuracy over 6 months
- **🎯 Real-time Analytics** with beautiful visualizations
- **💬 AI Chat Interface** (React only)
- **⚙️ Visual API Configuration** (React only)
- **📈 Advanced Calibration** and drift analysis

---

## 🚨 **Need Help?**

### Common Issues:
```bash
# If React won't start:
cd web_modern && rm -rf node_modules && npm install

# If backend won't start:
pip install -r api_backend/requirements.txt

# If no demo data:
python scripts/demo_mode.py enable
```

### Check Everything Works:
- **React**: http://localhost:3000
- **Streamlit**: http://localhost:8501  
- **API Docs**: http://localhost:8000/docs

---

## 🎯 **What's Next?**

1. **📱 Try React on mobile** - works perfectly!
2. **💬 Test the Chat interface** with your API keys
3. **📊 Explore Streamlit analytics** for deep insights
4. **🎮 Check out the rich demo data** - 6 months of evolution!

**🎉 You're all set! Choose your experience and dive in!** 