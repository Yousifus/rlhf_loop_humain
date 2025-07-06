# Tests & Validation

This directory contains test suites, validation scripts, and data integrity checks for the RLHF Loop system.

## 📁 Directory Contents

### Data Connection Tests
- **`test_data_connections.py`** - Validates data pipeline connectivity and integrity
  - Tests database connections
  - Validates data schema compliance
  - Checks data flow between components

### Reflection & Model Tests  
- **`test_reflection_data.py`** - Tests model reflection and introspection capabilities
  - Validates reflection data quality
  - Tests model self-assessment accuracy
  - Checks reflection pipeline integrity

## 🧪 Running Tests

### Full Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_data_connections.py
python tests/test_reflection_data.py
```

### Individual Tests
```bash
# Data connection validation
cd tests && python test_data_connections.py

# Reflection system validation  
cd tests && python test_reflection_data.py
```

## 📊 Test Coverage

### Data Pipeline Tests
- ✅ Database connectivity
- ✅ Data schema validation
- ✅ Pipeline data flow
- ✅ Error handling and recovery

### Model System Tests
- ✅ Reflection data quality
- ✅ Model introspection accuracy
- ✅ Self-assessment calibration
- ✅ Feedback loop integrity

## 🔧 Test Requirements

- **pytest** - Primary testing framework
- **numpy** - Numerical validation
- **pandas** - Data structure testing
- **All main dependencies** - From `requirements.txt`

## 🚀 Integration

Tests validate the entire RLHF system:
- **Data flows** (`../data/`)
- **Model performance** (`../models/`)
- **Interface functionality** (`../interface/`)
- **Pipeline integrity** (`../scripts/`)

## 📈 Continuous Integration

These tests are designed for:
- **Pre-deployment validation**
- **Regression testing** 
- **Data quality assurance**
- **Model performance monitoring** 