"""SWE015 - Real-Time Chat Moderation.

Run: python app.py
Download trained model files into a 'models' folder next to this file.
In the original Colab project, the saved files can also be loaded from
its parent directory. Use MODEL_DIR to specify a different location.
"""

import os
import re
import time
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch
from gensim.models import Word2Vec
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", APP_DIR / "models"))
if "MODEL_DIR" not in os.environ and not MODEL_DIR.is_dir():
    # During development, models are stored in the parent Google Drive folder.
    MODEL_DIR = APP_DIR.parent


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z0-9\s!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
OFFENSIVE_LEXICON = ("idiot", "stupid", "moron")


def normalize_chat(text):
    text = str(text).lower()
    text = re.sub(
        r"\b[a-z0-9]*[a-z][a-z0-9]*\b",
        lambda match: match.group().translate(LEET_MAP),
        text,
    )
    return re.sub(r"([a-z])\1{2,}", r"\1\1", text)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, letter_a in enumerate(a, 1):
        current = [i]
        for j, letter_b in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (letter_a != letter_b),
            ))
        previous = current
    return previous[-1]


def check_offensive_words(message):
    words = re.findall(r"\b[a-z]+\b", normalize_chat(message))
    matches = set()
    for word in words:
        for target in OFFENSIVE_LEXICON:
            maximum = 1 if len(target) >= 5 else 0
            if edit_distance(word, target) <= maximum:
                matches.add(target)
    return sorted(matches)


class ChatNeuralNetwork(torch.nn.Module):
    """Same feedforward architecture used for training in Colab."""

    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)


def load_models():
    """Load previously trained models; no training happens in this application."""
    needed = [
        "tfidf.joblib", "baseline_model.joblib", "chat_word2vec.model",
        "chat_neural_network.pt", "nn_scaler.joblib",
        "bert_dynamic_int8_state_dict.pt", "bert_finetuned/config.json",
    ]
    missing = [name for name in needed if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {MODEL_DIR}: {', '.join(missing)}. "
            "Download the trained files before starting the app."
        )

    tfidf = joblib.load(MODEL_DIR / "tfidf.joblib")
    baseline = joblib.load(MODEL_DIR / "baseline_model.joblib")
    word2vec = Word2Vec.load(str(MODEL_DIR / "chat_word2vec.model"))
    scaler = joblib.load(MODEL_DIR / "nn_scaler.joblib")

    network = ChatNeuralNetwork()
    network.load_state_dict(torch.load(
        MODEL_DIR / "chat_neural_network.pt",
        map_location="cpu",
        weights_only=True,
    ), strict=True)
    network.eval()

    bert_dir = MODEL_DIR / "bert_finetuned"
    tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
    bert = AutoModelForSequenceClassification.from_pretrained(
        str(bert_dir)
    ).cpu().float().eval()

    # Rebuild the exact INT8 architecture before loading its stored state dict.
    quantized = torch.ao.quantization.quantize_dynamic(
        bert, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
    )
    quantized.load_state_dict(torch.load(
        MODEL_DIR / "bert_dynamic_int8_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    ), strict=True)
    quantized.eval()
    bert = None
    return tfidf, baseline, word2vec, scaler, network, tokenizer, bert, quantized


torch.set_num_threads(min(4, os.cpu_count() or 1))
(tfidf, baseline_model, w2v_model, scaler, model_nn,
 bert_tokenizer, loaded_bert, quantized_bert) = load_models()

MODEL_CHOICES = [
    "Quantized BERT (INT8)",
    "Original BERT (FP32)",
    "Neural Network (Word2Vec)",
    "Classical NLP (TF-IDF)",
]


def text_to_vector(text):
    words = re.findall(r"\b[a-z]+\b", str(text))
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if not vectors:
        return np.zeros(100, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def predict_message(message, selected_model):
    """Return model label, model probability and measured end-to-end CPU time."""
    start = time.perf_counter()
    if selected_model == "Classical NLP (TF-IDF)":
        features = tfidf.transform([clean_text(message)])
        probabilities = baseline_model.predict_proba(features)[0]
    elif selected_model == "Neural Network (Word2Vec)":
        vector = text_to_vector(clean_text(message)).reshape(1, -1)
        features = scaler.transform(vector).astype(np.float32)
        with torch.inference_mode():
            logits = model_nn(torch.from_numpy(features))
            probabilities = torch.softmax(logits, dim=1)[0].numpy()
    else:
        inputs = bert_tokenizer(
            message, return_tensors="pt", truncation=True, max_length=128,
        )
             global loaded_bert

        if selected_model == "Original BERT (FP32)" and loaded_bert is None:
            loaded_bert = AutoModelForSequenceClassification.from_pretrained(
                str(MODEL_DIR / "bert_finetuned")
            ).cpu().float().eval()

        model = (
            quantized_bert if selected_model == "Quantized BERT (INT8)"
            else loaded_bert
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)[0].numpy()

    predicted_index = int(np.argmax(probabilities))
    predicted_label = int(baseline_model.classes_[predicted_index]) if selected_model == "Classical NLP (TF-IDF)" else predicted_index
    return predicted_label, float(probabilities[predicted_index]) * 100, (time.perf_counter() - start) * 1000


def mask_message(message):
    return "".join("*" if character.isalnum() else character for character in message)


def format_logs(logs):
    if not logs:
        return "No messages tested yet."
    return "\n".join(
        f"{index}. {item['model']}\n   {item['label']} | {item['latency']:.2f} ms"
        for index, item in enumerate(logs, 1)
    )


def moderate_chat(message, selected_model, history, logs):
    history = list(history or [])
    logs = list(logs or [])
    if not message.strip():
        return message, history, logs, format_logs(logs), "Please enter a message."

    prediction, confidence, latency = predict_message(message, selected_model)
    label = "Offensive" if prediction == 1 else "Clean"
    displayed = mask_message(message) if prediction == 1 else message
    detected_count = len(check_offensive_words(message))

    history.append({"role": "user", "content": displayed})
    response = (
        f"**Classification:** {label}\n\n"
        f"**Model confidence:** {confidence:.2f}%\n\n"
        f"**Response time:** {latency:.2f} ms\n\n"
        f"**Model:** {selected_model}"
    )
    if detected_count:
        response += f"\n\n**Rule-based matches detected:** {detected_count}"
    history.append({"role": "assistant", "content": response})
    logs.append({"model": selected_model, "label": label, "latency": latency})
    return "", history, logs, format_logs(logs), f"Last result: {label} | {confidence:.2f}% confidence"


def reset_chat():
    return "", [], [], "No messages tested yet.", "No classification yet."


with gr.Blocks(title="SWE015 - AI Chat Moderation") as demo:
    gr.Markdown("# AI Chat Moderation System")
    gr.Markdown(
        "Real-time message classification using Classical NLP, Neural Networks, "
        "BERT, and Quantized BERT."
    )
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Live Moderated Chat", height=430)
            message_input = gr.Textbox(
                label="Your Message", placeholder="Type a message to check...", lines=2,
            )
            with gr.Row():
                send_button = gr.Button("Check Message", variant="primary")
                clear_button = gr.Button("Clear Chat")
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=MODEL_CHOICES,
                value="Quantized BERT (INT8)",
                label="Select Model",
            )
            last_result = gr.Markdown("No classification yet.")
            gr.Markdown("### Response Time Log")
            timing_log = gr.Textbox(
                value="No messages tested yet.", lines=12,
                interactive=False, label="Message Processing History",
            )

    chat_logs = gr.State([])
    inputs = [message_input, model_selector, chatbot, chat_logs]
    outputs = [message_input, chatbot, chat_logs, timing_log, last_result]
    send_button.click(fn=moderate_chat, inputs=inputs, outputs=outputs)
    message_input.submit(fn=moderate_chat, inputs=inputs, outputs=outputs)
    clear_button.click(fn=reset_chat, inputs=[], outputs=outputs)


if __name__ == "__main__":
    share_link = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    demo.launch(share=share_link)
