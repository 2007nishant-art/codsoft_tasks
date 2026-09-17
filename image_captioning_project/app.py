import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🖼️",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🖼️ AI Image Captioning")
st.write(
    "Upload an image and let Artificial Intelligence "
    "generate a meaningful caption for it."
)


# --------------------------------------------------
# LOAD PRE-TRAINED BLIP MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


# Load model
processor, model = load_model()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# GENERATE CAPTION
# --------------------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Button
    if st.button("✨ Generate Caption"):

        with st.spinner("AI is analyzing the image..."):

            # Convert image into model input
            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            # Generate caption
            with torch.no_grad():

                output = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5
                )

            # Convert generated tokens into text
            caption = processor.decode(
                output[0],
                skip_special_tokens=True
            )

        # Display result
        st.success("Caption Generated!")

        st.subheader("🤖 AI Generated Caption")

        st.write(
            f"**{caption.capitalize()}**"
        )


# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📌 How It Works")

st.write(
    """
    1. The user uploads an image.
    
    2. The pre-trained BLIP vision-language model
       analyzes the image.
    
    3. The model extracts visual information from
       the image.
    
    4. A Transformer-based language component
       generates a natural-language caption.
    
    5. The generated caption is displayed to the user.
    """
)

st.info(
    "Technology Used: Python, PyTorch, Hugging Face Transformers, "
    "BLIP and Streamlit"
)