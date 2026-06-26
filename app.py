"""
Fake News Detection System
===========================
Main Gradio Application
Author: AI & ML Internship Project
Description: A machine learning-powered fake news detector with a beautiful UI
"""

# ── Standard library imports ──────────────────────────────────────────────────
import os
import pickle
import re

# ── Third-party imports ───────────────────────────────────────────────────────
import gradio as gr
import numpy as np

# ── Local imports ─────────────────────────────────────────────────────────────
from utils import clean_text, get_confidence_level, get_sample_news

# ── Load trained model and vectorizer ────────────────────────────────────────
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def load_model():
    """Load the trained ML model and TF-IDF vectorizer from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Load once at startup
model, vectorizer = load_model()

# ── Prediction function ───────────────────────────────────────────────────────
def predict_news(text: str):
    """
    Predict whether a news article is REAL or FAKE.

    Args:
        text (str): Raw news article text entered by the user.

    Returns:
        tuple: (result_html, confidence_html) — two HTML strings for the UI.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not text or not text.strip():
        return (
            '<div class="result-box error-box">⚠️ Please enter some news text to analyze.</div>',
            ""
        )

    if len(text.strip()) < 20:
        return (
            '<div class="result-box error-box">⚠️ Text is too short. Please enter at least 20 characters.</div>',
            ""
        )

    # ── Preprocess and vectorize ──────────────────────────────────────────────
    cleaned = clean_text(text)          # lowercase, remove punctuation, etc.
    features = vectorizer.transform([cleaned])  # convert text → TF-IDF vector

    # ── Model prediction ──────────────────────────────────────────────────────
    prediction = model.predict(features)[0]         # 0 = Fake, 1 = Real
    probabilities = model.predict_proba(features)[0]  # confidence scores

    # ── Interpret results ─────────────────────────────────────────────────────
    is_real = bool(prediction == 1)
    confidence = float(np.max(probabilities)) * 100   # e.g., 87.3
    label = "✅ REAL NEWS" if is_real else "🚨 FAKE NEWS"
    css_class = "real-box" if is_real else "fake-box"
    emoji = "✅" if is_real else "🚨"
    confidence_label = get_confidence_level(confidence)  # "High", "Medium", etc.

    # ── Build result HTML ─────────────────────────────────────────────────────
    result_html = f"""
    <div class="result-box {css_class}">
        <div class="result-emoji">{emoji}</div>
        <div class="result-label">{label}</div>
        <div class="result-sublabel">
            This article appears to be <strong>{'genuine' if is_real else 'fabricated or misleading'}</strong> news content.
        </div>
    </div>
    """

    # ── Build confidence HTML ─────────────────────────────────────────────────
    bar_color = "#22c55e" if is_real else "#ef4444"
    confidence_html = f"""
    <div class="confidence-box">
        <div class="confidence-header">
            <span>🎯 Model Confidence</span>
            <span class="confidence-percent">{confidence:.1f}%</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{confidence:.1f}%; background:{bar_color};"></div>
        </div>
        <div class="confidence-footer">
            Confidence Level: <strong>{confidence_label}</strong>
            &nbsp;|&nbsp; Real: {probabilities[1]*100:.1f}%
            &nbsp;|&nbsp; Fake: {probabilities[0]*100:.1f}%
        </div>
    </div>
    """

    return result_html, confidence_html


# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --real-green: #22c55e;
    --real-bg: #f0fdf4;
    --real-border: #86efac;
    --fake-red: #ef4444;
    --fake-bg: #fef2f2;
    --fake-border: #fca5a5;
    --error-bg: #fffbeb;
    --error-border: #fcd34d;
    --surface: #ffffff;
    --bg: #f8fafc;
    --text: #1e293b;
    --muted: #64748b;
    --radius: 16px;
}

/* ── Global reset ── */
body, .gradio-container { font-family: 'Inter', sans-serif !important; background: var(--bg) !important; color: var(--text); }

/* ── Header ── */
.header-banner {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    border-radius: var(--radius);
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(99,102,241,0.3);
}
.header-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-weight: 400;
}

/* ── Card wrapper ── */
.card {
    background: var(--surface);
    border-radius: var(--radius);
    border: 1px solid #e2e8f0;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Result boxes ── */
.result-box {
    border-radius: var(--radius);
    padding: 2rem 1.5rem;
    text-align: center;
    border: 2px solid;
    margin-top: 0.5rem;
}
.real-box { background: var(--real-bg); border-color: var(--real-border); }
.fake-box { background: var(--fake-bg); border-color: var(--fake-border); }
.error-box { background: var(--error-bg); border-color: var(--error-border); color: #92400e; }
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label { font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; }
.real-box .result-label { color: #16a34a; }
.fake-box .result-label { color: #dc2626; }
.result-sublabel { font-size: 0.95rem; color: var(--muted); }

/* ── Confidence box ── */
.confidence-box {
    background: var(--surface);
    border: 1px solid #e2e8f0;
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-top: 0.75rem;
}
.confidence-header {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
}
.confidence-percent { font-size: 1.2rem; color: var(--primary); font-weight: 700; }
.confidence-bar-bg {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 12px;
    overflow: hidden;
    margin-bottom: 0.6rem;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.6s ease;
}
.confidence-footer { font-size: 0.82rem; color: var(--muted); text-align: center; }

/* ── Sample buttons ── */
.sample-btn { font-size: 0.8rem !important; padding: 0.35rem 0.75rem !important; border-radius: 9999px !important; }

/* ── Footer (app info bar) ── */
.footer-text {
    text-align: center;
    color: var(--muted);
    font-size: 0.82rem;
    padding: 1.25rem 0 0.5rem 0;
    border-top: 1px solid #e2e8f0;
    margin-top: 1.5rem;
}

/* ── Developer footer ── */
.dev-footer {
    margin-top: 1.5rem;
    padding: 0;
}
.dev-footer-inner {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #f0fdf4 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.75rem 1.5rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 16px rgba(99,102,241,0.07);
    position: relative;
    overflow: hidden;
}
/* subtle top accent line */
.dev-footer-inner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
    border-radius: 16px 16px 0 0;
}
.dev-footer-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    margin: 0.45rem 0;
    flex-wrap: wrap;
}
.dev-footer-icon {
    font-size: 1.15rem;
    line-height: 1;
    flex-shrink: 0;
}
.dev-footer-label {
    font-size: 0.88rem;
    font-weight: 500;
    color: #334155;
    letter-spacing: 0.01em;
    line-height: 1.4;
}
.dev-footer-name {
    font-size: 1.05rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.01em;
}
.dev-footer-divider {
    width: 40px;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
    border-radius: 9999px;
    margin: 0.75rem auto 0.65rem;
    opacity: 0.5;
}
.dev-footer-badge {
    display: inline-block;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 9999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    color: #6366f1;
    font-weight: 500;
    margin-top: 0.6rem;
    letter-spacing: 0.02em;
}
/* Responsive: stack gracefully on small screens */
@media (max-width: 480px) {
    .dev-footer-inner { padding: 1.4rem 1rem 1.2rem; }
    .dev-footer-name  { font-size: 0.95rem; }
    .dev-footer-label { font-size: 0.82rem; }
}

/* ── Textarea polish ── */
textarea { border-radius: 12px !important; font-size: 0.95rem !important; }
"""

# ── Sample news texts ─────────────────────────────────────────────────────────
SAMPLES = get_sample_news()


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui():
    """Construct and return the Gradio Blocks UI."""

    with gr.Blocks(css=CUSTOM_CSS, title="Fake News Detector") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-banner">
            <div class="header-title">🔍 Fake News Detection System</div>
            <div class="header-subtitle">
                AI & Machine Learning · Powered by Scikit-learn · TF-IDF + Logistic Regression
            </div>
        </div>
        """)

        # ── Main content area ─────────────────────────────────────────────────
        with gr.Row():

            # Left column — input
            with gr.Column(scale=1):
                gr.HTML('<div class="card">')
                gr.Markdown("### 📝 Enter News Article")

                news_input = gr.Textbox(
                    label="",
                    placeholder="Paste or type a news headline or article here...",
                    lines=10,
                    max_lines=20,
                    show_label=False,
                )

                # Action buttons
                with gr.Row():
                    predict_btn = gr.Button("🔍 Analyze News", variant="primary", scale=3)
                    clear_btn  = gr.Button("🔄 Reset", variant="secondary", scale=1)

                # Sample news section
                gr.Markdown("#### 💡 Try a sample:")
                with gr.Row():
                    for sample in SAMPLES:
                        gr.Button(sample["label"], elem_classes=["sample-btn"]).click(
                            fn=lambda s=sample["text"]: s,
                            outputs=news_input
                        )
                gr.HTML('</div>')

            # Right column — output
            with gr.Column(scale=1):
                gr.HTML('<div class="card">')
                gr.Markdown("### 📊 Analysis Result")

                result_out     = gr.HTML(label="Prediction")
                confidence_out = gr.HTML(label="Confidence")

                # Info box
                gr.HTML("""
                <div style="margin-top:1rem; padding:1rem; background:#f1f5f9; border-radius:12px; font-size:0.85rem; color:#475569;">
                    <strong>ℹ️ How it works:</strong><br>
                    The model uses <strong>TF-IDF vectorization</strong> to convert text into numerical
                    features, then a <strong>Logistic Regression</strong> classifier predicts whether the
                    news is real or fake. Confidence shows the model's certainty.
                </div>
                """)
                gr.HTML('</div>')

        # ── App info footer (original) ─────────────────────────────────────────
        gr.HTML("""
        <div class="footer-text">
            🤖 Fake News Detection System &nbsp;|&nbsp;
            Built with Python, Scikit-learn &amp; Gradio &nbsp;|&nbsp;
            AI &amp; ML Internship Project 2026 &nbsp;|&nbsp;
            ⚠️ For educational purposes only
        </div>
        """)

        # ── Developer footer ───────────────────────────────────────────────────
        # Separated by <hr> as requested; contains developer credits,
        # internship details, and institution name — fully responsive.
        gr.HTML("""
        <hr style="
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 0.25rem 0 0;
        ">

        <div class="dev-footer">
            <div class="dev-footer-inner">

                <!-- Developer name -->
                <div class="dev-footer-row">
                    <span class="dev-footer-icon">👨‍💻</span>
                    <span class="dev-footer-label">Developed by</span>
                    <span class="dev-footer-name">Naman Kumar</span>
                </div>

                <!-- Thin gradient divider -->
                <div class="dev-footer-divider"></div>

                <!-- Internship line -->
                <div class="dev-footer-row">
                    <span class="dev-footer-icon">🎓</span>
                    <span class="dev-footer-label">AI &amp; Machine Learning Internship Project</span>
                </div>

                <!-- Institution line -->
                <div class="dev-footer-row">
                    <span class="dev-footer-icon">🏛</span>
                    <span class="dev-footer-label">Government Polytechnic West Champaran</span>
                </div>

                <!-- Decorative badge -->
                <div>
                    <span class="dev-footer-badge">✦ &nbsp;2026 &nbsp;·&nbsp; AI &amp; ML &nbsp;·&nbsp; Python &nbsp;✦</span>
                </div>

            </div>
        </div>
        """)

        # ── Event handlers ────────────────────────────────────────────────────
        predict_btn.click(
            fn=predict_news,
            inputs=[news_input],
            outputs=[result_out, confidence_out],
        )

        # Clear all fields on reset
        clear_btn.click(
            fn=lambda: ("", "", ""),
            outputs=[news_input, result_out, confidence_out],
        )

        # Allow pressing Enter to submit
        news_input.submit(
            fn=predict_news,
            inputs=[news_input],
            outputs=[result_out, confidence_out],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",   # Accept connections from all interfaces
        server_port=7860,         # Default Hugging Face Spaces port
        share=False,              # Set True to get a temporary public URL locally
        show_error=True,
    )
