import os
import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# settings
K_MODEL_ID = "LikoKIko/OpenCensor-H1-Mini"
K_MAX_LEN = 256
K_THRESHOLD = 0.17
K_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(max(1, os.cpu_count() or 1))

# load the model once when the app starts
tokenizer = AutoTokenizer.from_pretrained(K_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    K_MODEL_ID, num_labels=1
).to(K_DEVICE).eval()

# run one fake prediction so the model is fully loaded before the first real request
with torch.inference_mode():
    _warm = tokenizer("שלום", return_tensors="pt", padding="max_length",
                truncation=True, max_length=K_MAX_LEN).to(K_DEVICE)
    _ = torch.sigmoid(model(**_warm).logits).item()

def clean_text(text):
    # remove extra spaces from the text
    return re.sub(r"\s+", " ", str(text)).strip()

def predict(text):
    text = clean_text(text)

    if not text:
        return "Type something first."

    # convert text to numbers the model can read
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=K_MAX_LEN,
    ).to(K_DEVICE)

    # run the model and get a score between 0 and 1
    # 0 = definitely clean, 1 = definitely toxic
    with torch.inference_mode():
        score = torch.sigmoid(model(**inputs).logits).item()

    # compare score to threshold and decide toxic or clean
    # example: score=0.85, threshold=0.17 → label=1 (toxic)
    label = 1 if score >= K_THRESHOLD else 0

    return f"Prob: {score:.4f} | Label: {label} (cutoff={K_THRESHOLD})"

# build the UI
with gr.Blocks(title="Hebrew Profanity Detector") as demo:
    gr.Markdown("## Hebrew Profanity Detector\nEnter Hebrew text. Output: probability and label.")
    inp = gr.Textbox(lines=4, label="Hebrew text")
    out = gr.Textbox(label="Result", interactive=False)
    btn = gr.Button("Check")
    btn.click(predict, inputs=inp, outputs=out, api_name="/predict")

    # example texts the user can click to test the model
    gr.Examples(
        examples=[["זה דבר מצוין"], ["!יש לי חרא חבר"]],
        inputs=inp,
        outputs=out,
        fn=predict,
        cache_examples=False,
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)