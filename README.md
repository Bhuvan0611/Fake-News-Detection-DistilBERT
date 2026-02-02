
# 📰 Cross-Domain Fake News Detection using DistilBERT

```

╔══════════════════════════════════════════════════════════════╗
║   Robust Fake News Detection across multiple writing styles ║
║   using a fine-tuned DistilBERT Transformer model          ║
╚══════════════════════════════════════════════════════════════╝

```

---

## 🎯 Objective

Most fake-news models work only on the dataset they were trained on.  
This project demonstrates **cross-domain generalization** — the ability of a transformer model to detect fake news across **very different writing styles**.

---

## 🗂️ Datasets Used

| Dataset | Domain | Writing Style |
|---------|--------|----------------|
| **ISOT Fake News** | News articles | Long formal journalism |
| **GossipCop** | Celebrity headlines | Short sensational headlines |
| **LIAR Dataset** | Political statements | Medium factual claims |

All datasets were merged and standardized into:

```

text | label
0 = Fake
1 = Real

```

Total samples ≈ **80,000**

---

## 🧠 Methodology

```

Merge Datasets → Tokenize → Fine-Tune DistilBERT → Validate →
Domain Analysis → Calibration Metrics → Visual Artifacts

```

### Steps

1. Merge multi-domain datasets
2. Tokenize with DistilBERT (max length = 192)
3. Fine-tune for binary classification
4. Train/Validation split (90/10)
5. Domain-wise robustness testing
6. Advanced evaluation metrics

---

## ⚙️ Model Architecture

| Property | Value |
|----------|-------|
| Base Model | `distilbert-base-uncased` |
| Transformer Layers | 6 |
| Parameters | 66M |
| Output | Fake vs Real |

---

## 📊 Evaluation Metrics

- Accuracy
- Macro F1 Score
- Balanced Accuracy
- AUROC
- Expected Calibration Error (ECE)
- Brier Score
- Overconfidence
- Domain-wise analysis

---

## 📈 Results (Validation)

| Metric | Value |
|-------|-------|
| Accuracy | ~90% |
| Macro F1 | ~0.90 |
| AUROC | ~0.97 |
| Balanced Accuracy | ~0.90 |

### 🌍 Domain-wise Performance

| Domain | Accuracy |
|--------|----------|
| ISOT (News Articles) | ~99% |
| LIAR (Political Statements) | ~92% |
| GossipCop (Headlines) | ~80% |

✔️ Demonstrates strong cross-domain robustness.

---

## 🖼️ Artifacts

Available inside `/artifacts`:

- ROC Curve
- Confusion Matrix
- Summary metrics JSON

---

## 🤖 Trained Model Weights

GitHub does not allow large model files.  
Download the trained model here:

👉 **[Download Model](PASTE_DRIVE_LINK_HERE)](https://drive.google.com/drive/folders/1-DqrQWzvIlYnIyOK5iYNmTe7qXVrZBcD?usp=drive_link)**

Place inside:

```

model/distilbert_fake_news_model/

````

---

## ▶️ How to Use the Model

```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("model/distilbert_fake_news_model")
model = DistilBertForSequenceClassification.from_pretrained("model/distilbert_fake_news_model")

text = "Breaking news headline here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
prediction = model(**inputs).logits.argmax(dim=1).item()

print("Fake" if prediction == 0 else "Real")
````

---

## 📁 Repository Structure

```
notebook/      → Full Colab pipeline
artifacts/     → Evaluation plots and metrics
model/         → Place downloaded model here
README.md
```

---

## 🚀 Highlights

✔️ Cross-domain fake news detection
✔️ Transformer-based NLP model
✔️ Robustness & calibration analysis
✔️ Research-grade evaluation pipeline
✔️ Fully reproducible in one notebook

```


