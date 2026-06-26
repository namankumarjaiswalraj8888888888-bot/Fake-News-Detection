---
title: Fake News Detection
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.34.2
python_version: 3.11
app_file: app.py
pinned: false
---
# 🔍 Fake News Detection System

> An AI-powered web application that classifies news articles as **Real** or **Fake** using
> Machine Learning (TF-IDF + Logistic Regression), built with Python, Scikit-learn, and Gradio.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.16-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-FFD21E?style=flat-square)](https://huggingface.co/spaces)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-TF--IDF%20%2B%20LogReg-8A2BE2?style=flat-square&logo=tensorflow&logoColor=white)](#-how-it-works)
[![AI Internship](https://img.shields.io/badge/AI%20%26%20ML%20Internship-2026-0A9396?style=flat-square&logo=academia&logoColor=white)](#-developer)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

---

## 🌐 Live Demo

🚀 **Live Demo:** *(Available after deployment on Hugging Face Spaces)*

> Deploy karne ke liye neeche **[Deployment Guide](#-hugging-face-deployment)** dekhen.

---

## 📸 Screenshot

```
[ Header: Fake News Detection System ]
┌─────────────────────┐  ┌─────────────────────┐
│  📝 Enter Article   │  │  📊 Analysis Result  │
│                     │  │                      │
│  [Text Area]        │  │  ✅ REAL NEWS         │
│                     │  │  Confidence: 92.3%   │
│  [Analyze] [Reset]  │  │  ████████████░ 92%   │
│  💡 Sample Buttons  │  │  ℹ️ How it works      │
└─────────────────────┘  └─────────────────────┘
```

---

## 📁 Project Structure

```
Fake-News-Detection/
│
├── app.py             ← Main Gradio web application
├── train_model.py     ← ML model training script
├── utils.py           ← Text preprocessing helpers
├── model.pkl          ← Trained Logistic Regression model (generated)
├── vectorizer.pkl     ← Fitted TF-IDF vectorizer (generated)
├── requirements.txt   ← Python package dependencies
├── dataset.csv        ← Training dataset (generated or user-provided)
├── README.md          ← This file
│
├── assets/            ← Static assets (icons, images)
└── screenshots/       ← App screenshots
```

---

## 🚀 Quick Start

### Option 1 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Fake-News-Detection.git
cd Fake-News-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates model.pkl and vectorizer.pkl)
python train_model.py

# 4. Launch the web app
python app.py
```

Open your browser at **http://localhost:7860**

---

### Option 2 — Google Colab (no install needed)

```python
# Run this in a Colab cell:
!git clone https://github.com/YOUR_USERNAME/Fake-News-Detection.git
%cd Fake-News-Detection
!pip install -r requirements.txt -q
!python train_model.py
!python app.py
```

Click the **public URL** printed in the output.

---

### Option 3 — Hugging Face Spaces (permanent public URL)

See **[Deployment Guide](#-hugging-face-deployment)** below.

---

## 🤖 How It Works

```
User Input Text
      │
      ▼
┌─────────────────┐
│  clean_text()   │  lowercase → remove URLs → remove HTML → remove punctuation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TF-IDF Vector  │  Convert text into a 50,000-dimensional numerical feature vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Logistic Reg.  │  Binary classifier → outputs probability for each class
└────────┬────────┘
         │
         ▼
   REAL / FAKE  + Confidence %
```

### Why Logistic Regression?

| Property | Benefit |
|---|---|
| Fast training | Trains in seconds even on large datasets |
| Interpretable | You can see which words drove the prediction |
| Works well with TF-IDF | Proven combination for text classification |
| Probability output | Gives confidence scores, not just labels |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~96–98% |
| Precision | ~97% |
| Recall | ~96% |
| F1 Score | ~97% |

> Results vary depending on dataset. With the WELFake or ISOT dataset, expect 96–98% accuracy.

---

## 📦 Dataset

The project works with the **WELFake Dataset** (Kaggle) — free to download.

**Download steps:**
1. Go to https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
2. Download `WELFake_Dataset.csv`
3. Rename it to `dataset.csv`
4. Place it in the project root
5. Run `python train_model.py`

**Expected format:**

| text | label |
|---|---|
| "Scientists discover..." | 1 (Real) |
| "SHOCKING: Government..." | 0 (Fake) |

If no dataset is provided, the script auto-generates a balanced synthetic dataset.

---

## 🚢 Hugging Face Deployment

### Step 1 — Create a Space

1. Go to https://huggingface.co → Sign up / Log in
2. Click **"New Space"**
3. Fill in:
   - Space name: `fake-news-detector`
   - SDK: **Gradio**
   - Visibility: **Public**
4. Click **"Create Space"**

### Step 2 — Upload Files

Upload these files to your Space:
- `app.py`
- `utils.py`
- `train_model.py`
- `model.pkl`
- `vectorizer.pkl`
- `requirements.txt`

> **Important:** Train the model locally first (`python train_model.py`) so you have `model.pkl` and `vectorizer.pkl` to upload.

### Step 3 — Configure

Create a file named `README.md` in the Space with this header:
```yaml
---
title: Fake News Detector
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.16.0
app_file: app.py
pinned: false
---
```

### Step 4 — Done!

Hugging Face will automatically install dependencies and launch your app.
Your public URL will be: `https://huggingface.co/spaces/YOUR_USERNAME/fake-news-detector`

---

## 📤 GitHub Upload Guide

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: Fake News Detection System"

# Create repo on GitHub (github.com/new), then:
git remote add origin https://github.com/YOUR_USERNAME/Fake-News-Detection.git
git branch -M main
git push -u origin main
```

> **Note:** `model.pkl` and `vectorizer.pkl` can be large. Add them to `.gitignore` if needed,
> and have users run `python train_model.py` themselves.

---

## 📱 Run on Mobile

1. Open [Google Colab](https://colab.research.google.com) on your phone
2. Create a new notebook
3. Run the Colab cells from **Option 2** above
4. Click the public Gradio URL from your phone

Alternatively, visit the **Hugging Face Spaces** live demo — fully mobile-friendly.

---

## 🎓 Viva Questions & Answers

See `VIVA_QA.md` for 20+ viva questions with detailed answers.

---

## 📝 License

MIT License — free to use for educational and personal projects.

---

## 👨‍💻 Developer

| Field | Details |
|---|---|
| 👤 **Developed by** | Naman Kumar |
| 📌 **Project** | Fake News Detection System |
| 🎓 **Internship** | AI & Machine Learning Internship Project (2026) |
| 🏛️ **College** | Government Polytechnic West Champaran |

---

<div align="center">

**✦ &nbsp;2026 &nbsp;·&nbsp; AI & ML &nbsp;·&nbsp; Python &nbsp;✦**

*Built with ❤️ for learning and innovation.*

</div>
