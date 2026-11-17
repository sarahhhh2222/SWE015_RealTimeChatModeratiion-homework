# SWE015 - Real-Time Chat Moderation

## Project Overview
This project is an AI-powered chat moderation application
that detects offensive messages in real time.

The application classifies messages as Clean or Offensive
and automatically masks offensive content.

## Machine Learning Models
Four models were developed and compared:

1. Classical NLP - TF-IDF + Logistic Regression
2. Neural Network - Word2Vec + Feedforward Network
3. Original BERT - Fine-tuned Transformer
4. Quantized BERT - Dynamic INT8 Quantization

## Dataset
Jigsaw Toxic Comment Classification Dataset

Final test set: 15,958 comments.

## Application Features
- Real-time message classification
- Offensive message masking
- Model confidence scores
- Response time tracking
- Switching between four models
- Rule-based detection of suspicious words

## Final Results

| Model | Macro-F1 | Size (MB) | Latency (ms) |
|---|---|---|---|
| Classical NLP | 0.8553 | 1.34 | 2.28 |
| Neural Network | 0.7909 | 57.14 | 1.23 |
| Original BERT | 0.8972 | 418.35 | 210.72 |
| Quantized BERT | 0.8983 | 173.77 | 61.55 |

Latency was measured on an Intel Xeon CPU
at 2.00 GHz using 100 runs per model.

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Gensim
- Gradio


## Download Trained Models

The trained models are stored on Google Drive because
the weight files are too large for the GitHub repository.

Download the models here:

[SWE015 Trained Models](https://drive.google.com/file/d/1MGkORaPXc3IWRrf5bj3U-Rf-FqsrrSpi/view?usp=sharing)

After downloading:
1. Extract the ZIP file.
2. Place the extracted `models` folder in the project directory.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run the application using `python app.py`.
