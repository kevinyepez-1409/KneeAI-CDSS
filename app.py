"""KneeAI research prototype.

This application is for research demonstration only. It is not a clinically
validated diagnostic device and must not be used for patient management.
"""

from __future__ import annotations

import hashlib
import os
from math import log
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models, regularizers


IMG_SIZE = (300, 300)
BASE_PATH = Path(__file__).resolve().parent
MODEL_FILENAME = "kneeai_weights_final.weights.h5"

# The Streamlit app uses the separate deployment-reference checkpoint.
# This checkpoint is not the source of the archived historical 0.8219 result.
DEFAULT_DEPLOYMENT_SHA256 = (
    "49abd3fa257833176a4055f9f2c1a191"
    "69bd5e31dbc85f0067aef88399b49b5e"
)

# Optional overrides:
# - KNEEAI_MODEL_PATH: exact local path to the deployment checkpoint.
# - KNEEAI_EXPECTED_SHA256: expected hash for the selected checkpoint.
MODEL_PATH_OVERRIDE = os.getenv("KNEEAI_MODEL_PATH", "").strip()
EXPECTED_MODEL_SHA256 = os.getenv(
    "KNEEAI_EXPECTED_SHA256",
    DEFAULT_DEPLOYMENT_SHA256,
).strip().lower()

CLASS_NAMES_5 = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]
CLASS_NAMES_3 = ["Non-OA", "Mild–Moderate OA", "Severe OA"]

# Post-hoc illustrative threshold selected on the archived internal test curve.
# It is not a validated or transferable operating point.
ENTROPY_THRESHOLD = 0.6

L2_REG = 3.7e-3
DROPOUT_RATE = 0.49

# The archived Keras generator did not explicitly override interpolation and
# therefore used nearest-neighbor interpolation. This compatibility helper
# works across recent and older Pillow versions.
PIL_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def resolve_model_weights_path() -> Path:
    """Resolve the deployment checkpoint without silently renaming artifacts."""
    if MODEL_PATH_OVERRIDE:
        return Path(MODEL_PATH_OVERRIDE).expanduser().resolve()

    candidates = [
        BASE_PATH / "models" / MODEL_FILENAME,
        BASE_PATH / MODEL_FILENAME,  # Backward compatibility with older layout.
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Return the preferred documented location for the error message.
    return candidates[0]


MODEL_WEIGHTS_PATH = resolve_model_weights_path()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_model_architecture() -> tf.keras.Model:
    """Reconstruct the archived five-class EfficientNetB3 topology."""
    inputs = layers.Input(shape=(300, 300, 3), name="input_radiograph")
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )

    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = layers.BatchNormalization(name="bn_top_1")(x)
    x = layers.Dense(
        512,
        activation="swish",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_512",
    )(x)
    x = layers.BatchNormalization(name="bn_top_2")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)
    x = layers.Dense(
        256,
        activation="swish",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_256",
    )(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_2")(x)
    outputs = layers.Dense(5, activation="softmax", name="output")(x)

    return models.Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB3_5class",
    )


@st.cache_resource
def load_research_model() -> tf.keras.Model | None:
    """Load only the verified deployment-reference checkpoint."""
    if not MODEL_WEIGHTS_PATH.exists():
        st.error(
            "Deployment weights file not found. Expected one of:\n\n"
            f"- `{BASE_PATH / 'models' / MODEL_FILENAME}`\n"
            f"- `{BASE_PATH / MODEL_FILENAME}`\n\n"
            "Alternatively, set the `KNEEAI_MODEL_PATH` environment variable."
        )
        return None

    if EXPECTED_MODEL_SHA256:
        actual_hash = sha256_file(MODEL_WEIGHTS_PATH).lower()
        if actual_hash != EXPECTED_MODEL_SHA256:
            st.error(
                "Model SHA-256 mismatch. The application will not load an "
                "unverified checkpoint.\n\n"
                f"Expected: `{EXPECTED_MODEL_SHA256}`\n\n"
                f"Actual: `{actual_hash}`"
            )
            return None

    try:
        model = build_model_architecture()
        _ = model(
            np.zeros((1, 300, 300, 3), dtype=np.float32),
            training=False,
        )
        model.load_weights(str(MODEL_WEIGHTS_PATH))
        return model
    except Exception as error:
        st.error(f"Could not load the research model: {error}")
        return None


def collapse_5_to_3(probabilities_5: np.ndarray) -> np.ndarray:
    """Aggregate five KL probabilities into the common three-class space."""
    p5 = np.asarray(probabilities_5, dtype=np.float64)
    if p5.shape != (5,):
        raise ValueError(
            f"Expected five probabilities, received shape {p5.shape}."
        )

    p3 = np.array(
        [p5[0] + p5[1], p5[2] + p5[3], p5[4]],
        dtype=np.float64,
    )

    total = float(p3.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid probability vector.")

    return p3 / total


def map_five_class_argmax_to_three(
    probabilities_5: np.ndarray,
) -> tuple[int, int]:
    """Apply manuscript Rule A: five-class argmax, then KL-to-severity mapping."""
    p5 = np.asarray(probabilities_5, dtype=np.float64)
    if p5.shape != (5,):
        raise ValueError(
            f"Expected five probabilities, received shape {p5.shape}."
        )

    predicted_kl_index = int(np.argmax(p5))

    if predicted_kl_index in (0, 1):
        predicted_three_class_index = 0
    elif predicted_kl_index in (2, 3):
        predicted_three_class_index = 1
    else:
        predicted_three_class_index = 2

    return predicted_kl_index, predicted_three_class_index


def normalized_shannon_entropy(probabilities: np.ndarray) -> float:
    """Compute normalized entropy in the supplied probability space."""
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)) / log(probs.size))


def aggregated_gradcam(
    image_batch: np.ndarray,
    model: tf.keras.Model,
    three_class_index: int,
    last_conv_layer_name: str = "top_activation",
) -> np.ndarray:
    """Generate Grad-CAM for the selected aggregated three-class output."""
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        target_layer = model.get_layer("top_conv")

    grad_model = tf.keras.Model(
        model.inputs,
        [target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, probabilities_5 = grad_model(
            image_batch,
            training=False,
        )

        if three_class_index == 0:
            target = probabilities_5[:, 0] + probabilities_5[:, 1]
        elif three_class_index == 1:
            target = probabilities_5[:, 2] + probabilities_5[:, 3]
        elif three_class_index == 2:
            target = probabilities_5[:, 4]
        else:
            raise ValueError("three_class_index must be 0, 1, or 2.")

    gradients = tape.gradient(target, conv_outputs)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    activation = conv_outputs[0]
    heatmap = activation @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the displayed input image."""
    image_array = (
        np.asarray(
            image.resize(IMG_SIZE, resample=PIL_NEAREST),
            dtype=np.float32,
        )
        / 255.0
    )

    resized_heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE,
    ).numpy()[..., 0]

    colored = cm.get_cmap("jet")(resized_heatmap)[..., :3]
    return np.clip(0.65 * image_array + 0.35 * colored, 0.0, 1.0)


st.set_page_config(
    page_title="KneeAI — Research Prototype",
    page_icon="🩻",
    layout="wide",
)

st.title("KneeAI — KOA Severity Research Prototype")

st.warning(
    "Research demonstration only. This system is not clinically validated "
    "and must not be used for diagnosis, treatment, autonomous triage, or "
    "patient management."
)

st.caption(
    "The entropy threshold H=0.6 is a post-hoc illustrative setting selected "
    "on an archived internal test-set curve. It is not a transferable or "
    "validated operating point."
)

st.caption(
    "The application uses a separate deployment-reference checkpoint. "
    "It is not the source of the archived historical 0.8219 result reported "
    "from the canonical NPZ/CSV prediction artifacts."
)

model = load_research_model()

uploaded_file = st.file_uploader(
    "Upload an AP knee radiograph",
    type=["png", "jpg", "jpeg"],
)

if model is not None and uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as error:
        st.error(f"Could not read the uploaded image: {error}")
        st.stop()

    st.image(image, caption="Input radiograph", width=420)

    if st.button("Run research analysis", use_container_width=True):
        try:
            resized = image.resize(IMG_SIZE, resample=PIL_NEAREST)
            image_batch = np.expand_dims(
                np.asarray(resized, dtype=np.float32),
                axis=0,
            )
            image_batch = (
                tf.keras.applications.efficientnet.preprocess_input(
                    image_batch
                )
            )

            probabilities_5 = model.predict(image_batch, verbose=0)[0]
            probabilities_3 = collapse_5_to_3(probabilities_5)

            # Historical categorical Rule A:
            # five-class argmax followed by the predefined KL-to-severity map.
            predicted_kl_index, predicted_index = (
                map_five_class_argmax_to_three(probabilities_5)
            )
            predicted_label = CLASS_NAMES_3[predicted_index]
            predicted_kl_label = CLASS_NAMES_5[predicted_kl_index]

            # Probability-level analyses remain in the aggregated three-class
            # space, consistent with the manuscript.
            confidence = float(probabilities_3[predicted_index])
            entropy_3 = normalized_shannon_entropy(probabilities_3)
            is_ambiguous = entropy_3 > ENTROPY_THRESHOLD

            col1, col2, col3 = st.columns(3)
            col1.metric("Mapped severity category", predicted_label)
            col2.metric(
                "Aggregated probability for mapped category",
                f"{confidence:.2%}",
            )
            col3.metric(
                "Normalized three-class entropy",
                f"{entropy_3:.4f}",
            )

            st.caption(
                f"Internal five-class argmax: {predicted_kl_label}. "
                "The displayed category is obtained by mapping this KL argmax "
                "into the predefined three-class output space."
            )

            if is_ambiguous:
                st.error(
                    "Ambiguity flag: the output exceeds the illustrative "
                    "post-hoc entropy threshold. This is a research-prototype "
                    "flag, not a clinical recommendation."
                )
            else:
                st.info(
                    "The output does not exceed the illustrative entropy "
                    "threshold. This does not establish diagnostic certainty."
                )

            st.subheader("Five-grade internal probability profile")
            st.bar_chart(
                pd.DataFrame(
                    {"Probability": probabilities_5},
                    index=CLASS_NAMES_5,
                )
            )

            st.subheader("Aggregated three-class probability profile")
            st.bar_chart(
                pd.DataFrame(
                    {"Probability": probabilities_3},
                    index=CLASS_NAMES_3,
                )
            )

            heatmap = aggregated_gradcam(
                image_batch,
                model,
                predicted_index,
            )
            overlay = overlay_heatmap(image, heatmap)

            figure, axis = plt.subplots(figsize=(6, 6))
            axis.imshow(overlay)
            axis.axis("off")
            axis.set_title(
                "Grad-CAM visual plausibility map "
                f"for aggregated category: {predicted_label}"
            )
            st.pyplot(figure)
            plt.close(figure)

            st.caption(
                "Grad-CAM is a post-hoc visual plausibility tool. It is not "
                "causal evidence of model reasoning and does not replace "
                "expert review."
            )

        except Exception as error:
            st.error(f"Research analysis failed: {error}")
