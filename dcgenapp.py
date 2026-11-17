from pathlib import Path
from io import BytesIO
import zipfile

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
GENERATOR_PATH = BASE_DIR / "dcgan_generator.keras"
LATENT_DIM = 100  # Matches the uploaded generator's input shape: (None, 100)

st.set_page_config(page_title="DCGAN Image Generator", page_icon="🖼️", layout="wide")
st.title("🖼️ DCGAN Image Generator")
st.write("Generate 64 × 64 RGB images using the trained DCGAN generator.")


@st.cache_resource(show_spinner="Loading the trained generator...")
def load_generator(model_path: str):
    # compile=False avoids requiring the original training optimizer/loss setup.
    return tf.keras.models.load_model(model_path, compile=False)


def to_pil_image(array: np.ndarray) -> Image.Image:
    """Convert one tanh-normalized generator output [-1, 1] to an RGB image."""
    array = np.asarray(array, dtype=np.float32)
    array = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def make_zip(images: list[Image.Image]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate(images, start=1):
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")
            archive.writestr(f"generated_image_{index:02d}.png", image_buffer.getvalue())
    return buffer.getvalue()


if not GENERATOR_PATH.is_file():
    st.error(f"Generator model not found: {GENERATOR_PATH.name}")
    st.info("Upload dcgan_generator.keras to the same GitHub repository folder as app.py.")
    st.stop()

try:
    generator = load_generator(str(GENERATOR_PATH))
except Exception as exc:
    st.error("The generator model could not be loaded. Check the TensorFlow/Keras versions in requirements.txt.")
    st.exception(exc)
    st.stop()

# Validate the saved model rather than silently generating with the wrong latent size.
input_shape = generator.input_shape
if isinstance(input_shape, list):
    st.error("This app expects a single-input Keras generator model.")
    st.stop()
if len(input_shape) != 2 or input_shape[-1] != LATENT_DIM:
    st.error(f"Unexpected generator input shape: {input_shape}. Expected (None, {LATENT_DIM}).")
    st.stop()

with st.sidebar:
    st.header("Generation settings")
    image_count = st.slider("Number of images", min_value=1, max_value=16, value=4)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    generate = st.button("Generate images", type="primary", width="stretch")
    st.caption("The uploaded generator accepts 100-dimensional random noise and produces 64 × 64 RGB images.")

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if generate:
    try:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(0, 1, size=(int(image_count), LATENT_DIM)).astype(np.float32)
        with st.spinner("Generating images..."):
            generated = generator.predict(noise, verbose=0)
        if generated.ndim != 4 or generated.shape[-1] != 3:
            st.error(f"Unexpected generator output shape: {generated.shape}. Expected (N, 64, 64, 3).")
            st.stop()
        st.session_state.generated_images = [to_pil_image(item) for item in generated]
    except Exception as exc:
        st.error("Image generation failed.")
        st.exception(exc)

images = st.session_state.generated_images
if images:
    st.subheader(f"Generated images ({len(images)})")
    columns = st.columns(4)
    for index, image in enumerate(images):
        with columns[index % 4]:
            st.image(image, caption=f"Image {index + 1}", width="stretch")
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")
            st.download_button(
                label=f"Download image {index + 1}",
                data=image_buffer.getvalue(),
                file_name=f"generated_image_{index + 1:02d}.png",
                mime="image/png",
                key=f"download_{index}",
                width="stretch",
            )
    st.download_button(
        "Download all images (ZIP)",
        data=make_zip(images),
        file_name="dcgan_generated_images.zip",
        mime="application/zip",
        type="primary",
    )
else:
    st.info("Choose the number of images and click **Generate images** in the sidebar.")

with st.expander("Model information"):
    st.write(f"**Generator input shape:** `{generator.input_shape}`")
    st.write(f"**Generator output shape:** `{generator.output_shape}`")
    st.write("The discriminator file is included in the project archive for future training/evaluation. This inference app does not need to load it.")
