# PCL Detection — SemEval 2022 Task 4 Subtask 1

Binary classification of Patronising and Condescending Language (PCL) in English news paragraphs.

**GitHub Repository:** https://github.com/SwayamShah14/NLP-ss9323
**Leaderboard Name:** swam14

---

## Novel Approach

**RoBERTa-base + Imbalance Handling + Threshold Tuning**

Two motivated deviations from the RoBERTa-base baseline (F1=0.48):
1. **Imbalance handling** — we compare **Focal Loss** (γ=2, α=0.75) vs **Upsampling** (duplicate minority class ~9x) and pick the better strategy
2. **Threshold tuning** — grid-searches [0.30, 0.70] on dev set to maximise positive-class F1

---

## Repository Structure

```
├── BestModel/
│   └── pcl_detection_nb.ipynb    # Full notebook: Stages 2–5 (EDA → training → evaluation)
├── dev.txt                    # Dev set predictions
├── test.txt                   # Test set predictions
```

**NOTE**: _Models are not committed due to being nearly 1GB in size. These models can be trained on running the Jupyter Notebook in BestModel/pcl_detection_nb.ipynb. They can also be found on [Google Drive]{https://drive.google.com/drive/folders/178rdwKf4volLcjlIyCXqDbtKlMj5gI-8?usp=sharing}_


---

## Notebook Structure

tl;dr parts 3-5 consist of code for the novel model.

The notebook is structured as follows:
- Part 1: Setup
- Part 2: EDA
- Part 3: Setup of model components
- Part 4: Novel model training
- Part 5: Threshold tuning
- Part 6: Ablation study
- Part 7: Evaluation and error analysis
- Part 8: Creating prediction textfiles

---

## Data

All data files are downloaded automatically by the notebook — no manual setup required.

---

## Results

| Model | Dev F1 |
|---|---|
| RoBERTa-base (task baseline) | 0.48 |
| **RoBERTa-base + Upsampling + BCE + tuned threshold (BestModel)** | **0.6005** |

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- `transformers>=4.40.0`
- `datasets`, `accelerate`
- `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `nltk`
