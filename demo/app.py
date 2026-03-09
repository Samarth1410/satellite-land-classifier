import gradio as gr
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image

# ── Load model and processor ──────────────────────────────────────────
model_path = "../models/satellite-vit-final"

processor = ViTImageProcessor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

class_names = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
    "Industrial Buildings", "Pasture", "Permanent Crop",
    "Residential Buildings", "River", "SeaLake"
]

# ── Inference function ─────────────────────────────────────────────────
def classify(image: Image.Image):
    # Preprocess the image
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probs = outputs.logits.softmax(dim=-1)[0]

    # Return as a dictionary of {class_name: probability}
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# ── Gradio Interface ───────────────────────────────────────────────────
demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload Satellite Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🛰️ Satellite Land Use Classifier",
    description="""
    Upload a satellite image to classify its land use type.
    This model is a Vision Transformer (ViT) fine-tuned on the EuroSAT dataset.
    It can classify 10 categories: Annual Crop, Forest, Herbaceous Vegetation,
    Highway, Industrial Buildings, Pasture, Permanent Crop,
    Residential Buildings, River, and SeaLake.
    """,
    examples=[],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True generates a public link