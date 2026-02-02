title: "Cross-Domain Fake News Detection using DistilBERT"

overview:
  description: >
    A research-grade Machine Learning project that builds a robust Fake News
    Detection system using a fine-tuned DistilBERT transformer model trained
    across multiple domains of news data and evaluated with advanced robustness
    and calibration metrics.
  objective: >
    Demonstrate cross-domain generalization of a transformer-based NLP model
    across very different writing styles of news content.

datasets:
  - name: "ISOT Fake News"
    domain: "News articles"
    style: "Long formal journalism"
  - name: "GossipCop"
    domain: "Celebrity headlines"
    style: "Short sensational headlines"
  - name: "LIAR Dataset"
    domain: "Political statements"
    style: "Medium-length factual claims"
  format:
    columns: ["text", "label"]
    label_mapping:
      0: "Fake"
      1: "Real"
  total_samples: "~80,000"

methodology:
  steps:
    - "Merge multi-domain datasets into a unified dataset"
    - "Tokenize text using DistilBERT tokenizer (max length = 192)"
    - "Fine-tune DistilBERT for binary classification"
    - "Perform train/validation split (90/10)"
    - "Evaluate on unseen validation data"
    - "Perform domain-wise robustness analysis"
    - "Compute advanced evaluation metrics"
    - "Generate ROC curve and Confusion Matrix"

model:
  base_model: "distilbert-base-uncased"
  architecture:
    layers: 6
    parameters: "66M"
    head: "Binary classification (Fake vs Real)"

evaluation_metrics:
  - "Accuracy"
  - "Macro F1 Score"
  - "Balanced Accuracy"
  - "AUROC"
  - "Expected Calibration Error (ECE)"
  - "Brier Score"
  - "Overconfidence"
  - "Domain-wise performance"

results:
  validation:
    accuracy: "~90%"
    macro_f1: "~0.90"
    auroc: "~0.97"
    balanced_accuracy: "~0.90"
  domain_wise:
    - domain: "ISOT (News Articles)"
      accuracy: "~99%"
    - domain: "LIAR (Political Statements)"
      accuracy: "~92%"
    - domain: "GossipCop (Headlines)"
      accuracy: "~80%"

artifacts:
  folder: "artifacts/"
  files:
    - "roc_curve.png"
    - "confusion_matrix.png"
    - "summary.json"

trained_model:
  storage: "Google Drive"
  download_link: ["PASTE_DRIVE_LINK_HERE"](https://drive.google.com/drive/folders/1-DqrQWzvIlYnIyOK5iYNmTe7qXVrZBcD?usp=drive_link)
  placement: "model/distilbert_fake_news_model/"

usage_example:
  code: |
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    tokenizer = DistilBertTokenizerFast.from_pretrained("model/distilbert_fake_news_model")
    model = DistilBertForSequenceClassification.from_pretrained("model/distilbert_fake_news_model")

repository_structure:
  - "notebook/ : Colab notebook with full pipeline"
  - "artifacts/ : Evaluation plots and metrics"
  - "model/ : Place downloaded model here"
  - "README.md"

highlights:
  - "Cross-domain fake news detection"
  - "Transformer-based NLP model"
  - "Robustness and calibration analysis"
  - "Research-grade evaluation pipeline"
  - "Fully reproducible in a single notebook"
