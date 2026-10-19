
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def set_page_config():
    st.set_page_config(
        page_title="Caption an Image",
        page_icon="📷",
        layout="wide",
    )


@st.cache_resource
def initialize_model():
    hf_model = "Salesforce/blip-image-captioning-large"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(hf_model)
    model = BlipForConditionalGeneration.from_pretrained(
        hf_model
    ).to(device)

    model.eval()

    return processor, model, device


def upload_image():
    return st.sidebar.file_uploader(
        "Upload an image (we aren't storing anything)",
        type=["jpg", "jpeg", "png"],
    )


def resize_image(image, max_width):
    width, height = image.size

    if width > max_width:
        ratio = max_width / width
        height = int(height * ratio)
        image = image.resize((max_width, height))

    return image


def generate_caption(processor, model, device, image):
    inputs = processor(
        image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True,
    )

    return caption


def main():
    set_page_config()

    st.header("Caption an Image 📷")

    uploaded_image = upload_image()

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image = resize_image(image, max_width=300)

        st.image(image, caption="Your image")

        if st.sidebar.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                try:
                    processor, model, device = initialize_model()

                    caption = generate_caption(
                        processor,
                        model,
                        device,
                        image,
                    )

                    st.subheader("Caption:")
                    st.markdown(f"**{caption}**")

                except Exception as error:
                    st.error(f"Could not generate caption: {error}")


if __name__ == "__main__":
    main()


