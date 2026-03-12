import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="OliVidi — Olive Leaf Disease Detector",
    page_icon="🫒",
    layout="wide",
)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "olive_model.h5")


@st.cache_resource
def load_cnn_model():
    from olive_net import load_model
    return load_model()


def preprocess_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized


with st.sidebar:
    st.title("🫒 OliVidi")
    st.markdown("**Olive Leaf Disease Detection & Consultation**")
    st.markdown("---")
    st.markdown(
        """
        **How to use:**
        1. Upload an olive leaf photo
        2. View the CNN prediction
        3. Read the AI consultation report

        **Model:** OliNet CNN (Binary)
        **Classes:** Healthy / Diseased (Peacock Eye)
        **Agent:** Ollama (Llama 3) + ChromaDB
        """
    )
    st.markdown("---")
    st.caption("OliVidi — Powered by AI for Olive Health")

st.title("Olive Leaf Disease Detection")
st.markdown("Upload an olive leaf image to detect diseases and get expert treatment advice.")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found at `models/olive_model.h5`. "
        "Please run `python data_pipeline.py` then `python olive_net.py` first."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an olive leaf image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Take a clear photo of the leaf's upper surface for best results.",
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    uploaded_file.seek(0)
    img_array = preprocess_uploaded_image(uploaded_file)

    with st.spinner("Analyzing leaf..."):
        model = load_cnn_model()
        from olive_net import predict_from_array
        label, confidence = predict_from_array(img_array, model)

    with col2:
        st.subheader("Prediction Result")

        if label == "healthy":
            st.success(f"**{label.upper()}**", icon="✅")
        else:
            st.error(f"**{label.upper()}** — Olive Peacock Eye Spot", icon="🦚")

        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)

        st.markdown("---")
        st.info(
            "**Note:** This is an AI-assisted prediction. "
            "Always consult an agronomist for final diagnosis."
        )

    st.markdown("---")
    st.subheader("AI Consultation Report")

    with st.spinner("Generating consultation report via OliveConsultantAgent..."):
        try:
            from olive_agent import get_consultation
            report = get_consultation(label, confidence)
            st.markdown(report)
        except Exception as e:
            st.error(
                f"Could not generate consultation report: {e}\n\n"
                "Please check that your `docs/` folder contains knowledge base files."
            )
