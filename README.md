# Customer Service Chatbot Query Classification

Automated classification of banking customer service queries using machine learning. Achieves 94.7% accuracy across 10 issue categories.

## 🎯 Overview

Machine learning system that automatically routes customer queries to appropriate departments. Evaluates 13 different approaches from traditional ML to deep learning.

**Key Results:**
- 🥇 Best Model: Linear SVM (94.70% accuracy, 4.78% overfitting)
- 📊 Dataset: 2,169 queries, 10 balanced categories
- ⚡ Inference: <1ms per query
- 🚀 Production-ready

## 🏆 Model Performance

| Rank | Model | Test Acc | Gap | Inference | Status |
|------|-------|----------|-----|-----------|--------|
| 1 | **Linear SVM** | **94.70%** | **4.78%** | <1ms | ✅ |
| 2 | Linear SVM (Word+Char) | 94.47% | 5.36% | <1ms | ✅ |
| 3 | CNN for Text | 93.09% | 5.99% | 10ms | ✅ |
| 4 | Voting Ensemble | 93.09% | 6.39% | 5ms | ✅ |
| 5 | Logistic Regression | 92.17% | 6.68% | <1ms | ✅ |
| 6 | LSTM | 91.24% | 5.82% | 50ms | ⚠️ |
| 7 | Random Forest | 90.55% | 6.16% | 20ms | ⚠️ |
| 8 | XGBoost | 89.63% | 10.37% | 15ms | ❌ |

## 🔧 Technologies

- **ML:** scikit-learn, XGBoost, TensorFlow/Keras
- **NLP:** TF-IDF, word embeddings
- **Tools:** Python 3.8+, pandas, numpy, matplotlib

## 🚀 Quick Start
```bash
# Clone and install
git clone https://github.com/yourusername/Customer-Service-Chatbot-Query-Classification.git
cd Customer-Service-Chatbot-Query-Classification
pip install -r requirements.txt

# Run analysis
python notebooks/01_eda_analysis.py          # Data exploration
python notebooks/04_comprehensive_comparison.py  # Compare all models
python notebooks/05_deep_learning.py         # Deep learning (optional)
```

## 📊 Key Findings

### Why Linear SVM Won

**Success Factors:**
- ✅ Small dataset (2,169) - SVM excels with limited data
- ✅ Short text (~15 words) - keyword-based discrimination
- ✅ Sparse TF-IDF features - SVM's strength
- ✅ Fast & interpretable - <1ms inference, explainable

**Why Deep Learning Failed:**
- ❌ Insufficient data (692K params, 2K samples)
- ❌ Text too short for sequential modeling
- ❌ Simple keywords vs. complex semantics
- ❌ Overfitting (99% train, 93% test)

**Why Trees Failed:**
- ❌ Severe overfitting (100% train, 89% test)
- ❌ Poor with sparse high-dimensional features

## 🎨 10 Issue Categories

Card Payment Fee | Direct Debit Not Recognised | Balance Not Updated (Cheque) | Wrong Cash Amount | Withdrawal Charge | Transaction Charged Twice | Declined Withdrawal | Transfer Fee | Transfer Not Received | Balance Not Updated (Transfer)

## 📈 Dataset

- **Size:** 2,169 queries
- **Split:** 80/20 train/test
- **Classes:** 10 (balanced, 1.08:1 ratio)
- **Text:** 15-409 chars (76 avg), 15 words avg
- **Quality:** No missing values, no duplicates

## 🔮 Future Improvements

1. Collect more data (5K+) → enable deep learning
2. Add domain-specific financial features
3. Ensemble SVM + neural network predictions
4. Target underperforming classes (7, 9)
5. Continuous learning pipeline



⭐ **Star this repo if you found it helpful!**
