import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from math import log
import os

# =========================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS
# =========================================================
st.set_page_config(
    page_title="KneeAI – Clinical CDSS",
    page_icon="🩺",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .section-card {
        background-color: #161b22;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #30363d;
        margin-bottom: 1.5rem;
    }
    .kneeai-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #58a6ff;
        margin-bottom: 0.2rem;
    }
    .kneeai-subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
    }
    .metric-label { color: #8b949e; font-size: 0.9rem; font-weight: 600; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
    .clinical-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f85149;
        background-color: rgba(248, 81, 73, 0.1);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 2. CONFIGURACIÓN GLOBAL DE RUTAS
# =========================================================
IMG_SIZE = (300, 300)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.join(BASE_PATH, "kneeai_weights_final.weights.h5")

CLASS_NAMES_5 = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]
CLASS_NAMES_3 = ["Non-OA", "Mild-Mod", "Severe"]
ENTROPY_THRESHOLD = 0.6

# =========================================================
# 3. CONSTRUCCIÓN DEL MODELO
# =========================================================
def build_model_architecture():
    inputs = tf.keras.Input(shape=(300, 300, 3), name="input_radiograph")
    base = tf.keras.applications.EfficientNetB3(include_top=False, weights=None, input_tensor=inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    x = tf.keras.layers.BatchNormalization(name="bn_top_1")(x)
    x = tf.keras.layers.Dense(512, activation='swish', name='dense_512_refined')(x)
    x = tf.keras.layers.BatchNormalization(name="bn_top_2")(x)
    x = tf.keras.layers.Dropout(0.49)(x)
    x = tf.keras.layers.Dense(256, activation='swish', name='dense_256_refined')(x)
    x = tf.keras.layers.Dropout(0.49)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax', name='kl_output')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

@st.cache_resource
def load_clinical_system():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        st.error(f"Weights file not found at: {MODEL_WEIGHTS_PATH}")
        return None
    try:
        model = build_model_architecture()
        dummy_input = np.zeros((1, 300, 300, 3))
        _ = model(dummy_input, training=False)
        model.load_weights(MODEL_WEIGHTS_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None

# =========================================================
# 4. FUNCIONES AUXILIARES
# =========================================================
def get_uncertainty(probs):
    return -np.sum(probs * np.log(probs + 1e-12)) / log(5)

def collapse_5_to_3(p5):
    return np.array([p5[0] + p5[1], p5[2] + p5[3], p5[4]])

def make_gradcam(img_array, model, last_conv_layer_name="top_activation"):
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        target_layer = model.get_layer("top_conv")
    grad_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# =========================================================
# 5. SIDEBAR Y LAYOUT
# =========================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=70)
    st.title("KneeAI 🩺")
    st.info("**CDSS Prototype**\n- Backbone: EfficientNetB3\n- Accuracy: 82.21%\n- Safety Filter Active")
    st.markdown(f"**Uncertainty Gate:** H < {ENTROPY_THRESHOLD}")
    st.caption("Clinical Research Use Only v2.3")

st.markdown('<div class="kneeai-title">KneeAI – Clinical Decision Support</div>', unsafe_allow_html=True)
st.markdown('<div class="kneeai-subtitle">Validated CDSS for Knee Osteoarthritis Severity Assessment</div>', unsafe_allow_html=True)

model = load_clinical_system()

if model:
    st.markdown('### 1. Patient Radiograph Acquisition')
    uploaded_file = st.file_uploader("Upload AP Knee X-ray", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img_raw = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown('<div class="section-card"><div class="section-header">Input Radiograph</div>', unsafe_allow_html=True)
            st.image(img_raw, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-card"><div class="section-header">Diagnostic Pipeline</div>', unsafe_allow_html=True)
            if st.button("🧠 START AI ANALYSIS", use_container_width=True):
                with st.spinner("Analyzing..."):
                    img_res = img_raw.resize(IMG_SIZE)
                    x = np.expand_dims(np.array(img_res), axis=0)
                    x = tf.keras.applications.efficientnet.preprocess_input(x)
                    probs_5 = model.predict(x, verbose=0)[0]
                    entropy_val = get_uncertainty(probs_5)
                    probs_3 = collapse_5_to_3(probs_5)
                    label_3 = CLASS_NAMES_3[np.argmax(probs_3)]
                    confidence = probs_3[np.argmax(probs_3)]
                    heatmap = make_gradcam(x, model)

                    st.markdown('<div class="section-header" style="margin-top:1rem;">3. Clinical Findings</div>', unsafe_allow_html=True)
                    if entropy_val >= ENTROPY_THRESHOLD:
                        st.markdown(f'<div class="clinical-alert"><strong>🚨 AMBIGUITY ALERT (H={entropy_val:.2f})</strong><br>Features are inconsistent. Manual review required.</div>', unsafe_allow_html=True)
                    else:
                        c_res1, c_res2 = st.columns(2)
                        with c_res1:
                            color = "#f85149" if label_3 == "Severe" else "#d29922" if label_3 == "Mild-Mod" else "#3fb950"
                            st.markdown(f'<div class="metric-label">Diagnosis</div><div class="metric-value" style="color:{color}">{label_3}</div>', unsafe_allow_html=True)
                        with c_res2:
                            st.markdown(f'<div class="metric-label">Confidence</div><div class="metric-value">{confidence:.2%}</div>', unsafe_allow_html=True)
                        st.markdown("**Management Suggestion:**")
                        if label_3 == "Non-OA": st.success("Routine monitoring and prevention.")
                        elif label_3 == "Mild-Mod": st.warning("Conservative management advised.")
                        else: st.error("Urgent surgical evaluation referral.")
            st.markdown('</div>', unsafe_allow_html=True)

        if 'entropy_val' in locals():
            t1, t2 = st.columns(2)
            with t1:
                st.markdown('<div class="section-card"><div class="section-header">4. Uncertainty & KL Profile</div>', unsafe_allow_html=True)
                st.write(f"Shannon Entropy: **{entropy_val:.4f}**")
                st.progress(min(float(entropy_val), 1.0))
                st.bar_chart(pd.DataFrame(probs_5, index=CLASS_NAMES_5, columns=["Prob"]))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with t2:
                st.markdown('<div class="section-card"><div class="section-header">5. Visual Explainability</div>', unsafe_allow_html=True)
                h_res = cv2.resize(heatmap, (IMG_SIZE[0], IMG_SIZE[1]))
                h_col = cv2.applyColorMap(np.uint8(255 * h_res), cv2.COLORMAP_JET)
                h_col = cv2.cvtColor(h_col, cv2.COLOR_BGR2RGB)
                super_img = cv2.addWeighted(np.array(img_res), 0.6, h_col, 0.4, 0)
                st.image(super_img, caption="Biological markers detected (Grad-CAM)", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card"><div class="section-header">6. Clinical Risk Radar</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
            angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
            vals = probs_3.tolist(); vals += vals[:1]; angles += angles[:1]
            ax.fill(angles, vals, color='#58a6ff', alpha=0.3)
            ax.plot(angles, vals, color='#58a6ff', linewidth=3)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(CLASS_NAMES_3, color="#8b949e", size=11)
            ax.set_yticklabels([]); plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; color:#6b7280; font-size:0.8rem; margin-top:2rem;">KneeAI Framework · CDSS Support Tools</div>', unsafe_allow_html=True)