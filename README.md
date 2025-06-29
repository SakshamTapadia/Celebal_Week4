# Spam Email Detection System

A **production-grade machine learning system** to classify spam vs ham emails using classical ML and deep learning models. Built with modularity, scalability, and best practices in mind.

## Features

- Modular architecture with clean code separation
- Configuration-driven pipeline (YAML)
- Multiple models: Logistic Regression, SVM, Random Forest, XGBoost, LSTM, CNN, BERT
- Cross-validation and Optuna-based hyperparameter tuning
- Rich evaluation (AUC, confusion matrix, ROC, PR curve)
- Interpretability with SHAP (optional)
- Containerized with Docker
- CLI + API-ready prediction layer
- Unit test-ready and extensible

## Project Structure

```
├── config/
│   └── config.yaml
├── data/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_extractor.py
│   └── data_splitter.py
├── models/
│   ├── base_model.py
│   ├── traditional_models.py
│   ├── deep_learning_models.py
│   ├── ensemble_models.py
│   └── model_selector.py
├── evaluation/
│   ├── metrics.py
│   ├── evaluator.py
│   └── visualizer.py
├── utils/
│   ├── config.py
│   ├── logger.py
│   └── helpers.py
├── main.py
├── train.py
├── predict.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup Instructions

```bash
# Clone repo and install dependencies
git clone https://github.com/yourname/spam-detector.git
cd spam-detector
pip install -r requirements.txt
```

## Training a Model

```bash
python train.py
```

Trains model based on config in `config/config.yaml` and saves artifacts in `artifacts/`.

## Making Predictions

```bash
python predict.py "Win a free iPhone by replying now!"
# Output: SPAM
```

## Docker Support

```bash
docker build -t spam-detector .
docker run --rm spam-detector "This is a limited-time offer!"
```

## Configuration

Modify `config/config.yaml`:

```yaml
model:
  type: "XGBoost"
  parameters:
    n_estimators: 100
    max_depth: 5
features:
  tfidf: true
  ngram_range: [1, 2]
```

## Evaluation Metrics

After training:
- Precision, Recall, F1, AUC
- Confusion Matrix Heatmap
- ROC and PR Curves

## Future Improvements

- [ ] Add REST API via FastAPI
- [ ] Integrate MLFlow for experiment tracking

## Maintainer

Built by Saksham Tapadia
License: MIT
