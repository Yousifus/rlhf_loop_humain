# 🚀 Deployment Guide - RLHF Loop System

*Enterprise-grade deployment instructions for production environments*

---

## 📋 **Prerequisites**

### 🖥️ **System Requirements**
- **CPU:** 4+ cores, 2.5GHz minimum (8+ cores recommended)
- **Memory:** 8GB RAM minimum (16GB+ for production)
- **Storage:** 50GB+ free space (SSD recommended)
- **OS:** Windows 10+, macOS 10.15+, or Ubuntu 20.04+

### 🐍 **Software Dependencies**
```bash
Python 3.8+          # Core runtime
Node.js 16+          # Frontend tooling
Git                  # Version control
PowerShell 5.1+      # Windows automation (Windows only)
```

---

## ⚡ **Quick Deployment**

### 🚀 **Option 1: Automated Setup**
```bash
# Clone and deploy in one command
git clone https://github.com/Yousifus/rlhf_loop_humain.git
cd rlhf_loop_humain
./scripts/deploy.sh  # Linux/macOS
# OR
scripts/deploy.ps1   # Windows
```

### 🛠️ **Option 2: Manual Setup**
```bash
# 1. Clone repository
git clone https://github.com/Yousifus/rlhf_loop_humain.git
cd rlhf_loop_humain

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Node.js dependencies (if using web components)
npm install

# 4. Setup configuration
cp config/.taskmasterconfig.example config/.taskmasterconfig

# 5. Launch dashboard
python scripts/run_enhanced_dashboard_v2.py
```

---

## 🐳 **Docker Deployment**

### 📦 **Containerized Setup**
```dockerfile
# Dockerfile (example configuration)
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "scripts/run_enhanced_dashboard_v2.py"]
```

### 🚀 **Docker Commands**
```bash
# Build image
docker build -t rlhf-loop-humain .

# Run container
docker run -p 8501:8501 rlhf-loop-humain

# Docker Compose (for full stack)
docker-compose up -d
```

---

## ☁️ **Cloud Deployment**

### 🌐 **AWS Deployment**
```yaml
# AWS Elastic Beanstalk configuration
platform: python-3.8
services:
  - name: rlhf-dashboard
    type: web
    instance_type: t3.medium
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - ENVIRONMENT=production
```

### 🔵 **Azure Deployment**
```bash
# Azure Container Instances
az container create \
  --resource-group rlhf-rg \
  --name rlhf-dashboard \
  --image rlhf-loop-humain:latest \
  --ports 8501 \
  --cpu 2 --memory 4
```

### 🟢 **Google Cloud Deployment**
```yaml
# Google Cloud Run service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: rlhf-dashboard
spec:
  template:
    spec:
      containers:
      - image: gcr.io/project/rlhf-loop-humain
        ports:
        - containerPort: 8501
```

---

## 🔧 **Configuration Management**

### ⚙️ **Environment Variables**
```bash
# Core settings
ENVIRONMENT=production
LOG_LEVEL=info
DEBUG_MODE=false

# API Configuration
DEEPSEEK_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
LMSTUDIO_API_BASE=http://localhost:1234/v1

# Database Settings
DATABASE_URL=postgresql://user:pass@host:5432/rlhf
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
```

### 📁 **Configuration Files**
```bash
config/
├── .taskmasterconfig       # Task management settings
├── database.conf           # Database configuration
├── api_keys.conf           # API key management (local only)
└── deployment.conf         # Environment-specific settings
```

---

## 🛡️ **Security Configuration**

### 🔒 **SSL/TLS Setup**
```nginx
# Nginx configuration (recommended)
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 👥 **Access Control**
```python
# Authentication middleware (example)
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'oauth2_provider.backends.OAuth2Backend',
]

# Role-based permissions
ROLES = {
    'admin': ['read', 'write', 'delete'],
    'analyst': ['read', 'write'],
    'viewer': ['read']
}
```

---

## 📊 **Monitoring & Logging**

### 📈 **Application Monitoring**
```yaml
# Prometheus configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rlhf-dashboard'
    static_configs:
      - targets: ['localhost:8501']
    metrics_path: /metrics
```

### 📝 **Logging Configuration**
```python
# logging.conf
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '{levelname} {asctime} {name} {process:d} {thread:d} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs/rlhf_system.log',
            'formatter': 'standard',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file'],
    },
}
```

---

## 🔄 **CI/CD Pipeline**

### 🏗️ **GitHub Actions**
```yaml
# .github/workflows/deploy.yml
name: Deploy RLHF System

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        python -m pytest tests/
        
    - name: Deploy to production
      run: |
        ./scripts/deploy.sh
```

---

## 🧪 **Health Checks**

### ✅ **System Validation**
```bash
# Health check endpoints
curl http://localhost:8501/health          # Basic health check
curl http://localhost:8501/metrics         # Performance metrics
curl http://localhost:8501/ready           # Readiness probe
curl http://localhost:8501/live            # Liveness probe
```

### 🔍 **Troubleshooting**
```bash
# Check system status
python scripts/system_check.py

# Validate configuration
python scripts/validate_config.py

# Test database connections
python tests/test_data_connections.py

# Monitor logs
tail -f logs/rlhf_system.log
```

---

## 📈 **Performance Optimization**

### ⚡ **Production Tuning**
```python
# Streamlit configuration
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#2563eb"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
```

### 🔄 **Scaling Strategies**
- **Horizontal Scaling:** Multiple container instances with load balancer
- **Vertical Scaling:** Increased CPU/memory allocation
- **Database Optimization:** Connection pooling and query optimization
- **Caching Layer:** Redis for frequently accessed data

---

## 🚨 **Backup & Recovery**

### 💾 **Data Backup**
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/rlhf-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configuration
cp -r config/ $BACKUP_DIR/

# Backup data
pg_dump rlhf_database > $BACKUP_DIR/database.sql

# Backup models
tar -czf $BACKUP_DIR/models.tar.gz models/
```

### 🔄 **Recovery Procedures**
```bash
# Database recovery
psql rlhf_database < backup/database.sql

# Model recovery
tar -xzf backup/models.tar.gz -C models/

# Configuration recovery
cp -r backup/config/* config/
```

---

<div align="center">

## 🌟 **Ready for Enterprise Production** 🌟

*Comprehensive deployment guide for mission-critical applications*

</div> 