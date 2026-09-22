"""
Streamlit web application for Satellite Image Land-Cover Classification.

Upload a satellite image -> get predicted class, confidence, full probability
distribution, and a Grad-CAM explainability visualization.
"""

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import ModelPredictor
from src.explainability.gradcam import GradCAM, overlay_heatmap

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

CLASS_COLORS = {
    "AnnualCrop": "#D4A24C",
    "Forest": "#2E7D32",
    "HerbaceousVegetation": "#7CB342",
    "Highway": "#616161",
    "Industrial": "#8D6E63",
    "Pasture": "#9CCC65",
    "PermanentCrop": "#C77B00",
    "Residential": "#EF5350",
    "River": "#29B6F6",
    "SeaLake": "#0D47A1",
}

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        margin-bottom: 1.2rem;
    }
    .prediction-class {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .prediction-confidence {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.55rem;
    }
    .prob-label {
        width: 150px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .prob-bar-bg {
        flex-grow: 1;
        background-color: #eee;
        border-radius: 6px;
        height: 18px;
        overflow: hidden;
        margin-right: 0.6rem;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
    }
    .prob-value {
        width: 55px;
        font-size: 0.85rem;
        text-align: right;
        color: #444;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.8rem;
        margin-bottom: 0.6rem;
    }
    .caption-box {
        background-color: #f5f7fa;
        border-left: 4px solid #2a5298;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #444;
        margin-bottom: 1rem;
    }
</style>
"""


@st.cache_resource
def load_predictor():
    return ModelPredictor()


def denormalize_tensor(tensor):
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0, 1)


def render_probability_bars(probabilities: dict):
    sorted_probs = sorted(probabilities.items(), key=lambda x: -x[1])
    for cls, prob in sorted_probs:
        color = CLASS_COLORS.get(cls, "#2a5298")
        pct = prob * 100
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label">{cls}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%; background-color:{color};"></div>
                </div>
                <div class="prob-value">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="Satellite Land-Cover Classification",
        page_icon="🛰️",
        layout="centered",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🛰️ About")
        st.write(
            "Deep learning land-cover classifier trained on the **EuroSAT** "
            "dataset (Sentinel-2 satellite imagery, 10 classes, 27,000 images)."
        )
        st.markdown("**Model:** Fine-tuned ResNet18")
        st.markdown("**Test Accuracy:** 97.93%")
        st.markdown("**Macro F1:** 0.9787")
        st.markdown("---")
        st.markdown("### Classes")
        for cls, color in CLASS_COLORS.items():
            st.markdown(
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'background-color:{color};border-radius:50%;margin-right:6px;"></span>{cls}',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="main-header">Satellite Image Land-Cover Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a satellite image tile to classify its land-cover type.</div>',
        unsafe_allow_html=True,
    )

    predictor = load_predictor()

    uploaded_file = st.file_uploader(
        "Drag and drop or browse for an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("👆 Upload a satellite image to get started.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        result = predictor.predict(image)

    color = CLASS_COLORS.get(result["class"], "#2a5298")
    with col2:
        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="font-size:0.95rem; opacity:0.85;">PREDICTED CLASS</div>
                <div class="prediction-class">{result['class']}</div>
                <div class="prediction-confidence">Confidence: {result['confidence']:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)
    render_probability_bars(result["probabilities"])

    st.markdown('<div class="section-title">Explainability (Grad-CAM)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="caption-box">Grad-CAM highlights image regions that influenced the model\'s '
        'prediction. This is an approximate visualization of influential regions, not proof of the '
        'model\'s reasoning.</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Generating Grad-CAM visualization..."):
        input_tensor = predictor.transform(image).unsqueeze(0).to(predictor.device)
        gradcam = GradCAM(predictor.model, target_layer=predictor.model.layer4)
        heatmap, pred_idx, _ = gradcam.generate(input_tensor)

        img_np = denormalize_tensor(input_tensor.squeeze(0))
        overlay = overlay_heatmap(img_np, heatmap)

    col3, col4 = st.columns(2)
    with col3:
        st.image(img_np, caption="Original (preprocessed)", use_container_width=True)
    with col4:
        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)


if __name__ == "__main__":
    main()