# Sentiment Analysis
Sentiment analysis in Python for complaint cases data

## Complaint cases

- data format
- fields available
    - unstructured free-text
    - reporter attributes
    - company attributes

## 00 Data cleanup
cleanup and process data into format ready for analysis

cleanup tasks
- concat multiple Excel sheets by year into a single table with added column `year`

## 01 Sentiment
estimate sentiment from unstructured data from complaint cases

__options__

01. Lexicon-based approaches
Use emotion/valence/arousal lexicons where words are pre-scored on emotional dimensions.

- Advantages: Transparent, interpretable (easy to explain to regulators).
- Limitations: Misses nuance, context, sarcasm, new words.

Examples:

    - NRC Emotion Lexicon (EmoLex) → tags words with emotions like anger, fear, sadness.
    - ANEW / Warriner et al. norms → provide valence (pleasantness), arousal (calm–excited), dominance scores for thousands of English words.
    - LIWC (Linguistic Inquiry and Word Count) → commercial, widely used in psychology.

02. Supervised training model: custom solution
03. Transformer based emotion analysis (zero shot)

    - Use large language models or pretrained transformers fine-tuned on emotion/arousal datasets:
    - You can score text for multiple emotions, then map to arousal/intensity (e.g., anger > high arousal; sadness > lower arousal).

 - Advantages: No need for much local training, good with nuance.
 - Limitations: Pretraining on general data → cultural/legal context mismatch (Singlish, local complaint phrasing).

Examples:

- **GoEmotions (Google)** → 58 fine-grained emotions (anger, frustration, disappointment, etc.).
- **Hugging Face models**: joeddav/distilbert-base-uncased-go-emotions-student, cardiffnlp/twitter-roberta-base-emotion.

04. Valence-Arousal
Instead of "positive/negative," score along 2D space:

- Valence: unpleasant ↔ pleasant.
- Arousal: calm ↔ activated.

Each complaint can then be placed in that space.
Useful for regulators: e.g., “Angry and urgent complaints” vs. “Calm but dissatisfied complaints.”

Mapping options:
 - Use lexicon scores (ANEW/Warriner).
 - Train regressors with embeddings → predict arousal/intensity scores.

### Valence-Arousal

- download the Warriner dataset from [this url](https://github.com/JULIELab/XANEW/blob/master/Ratings_Warriner_et_al.csv) and save it to local CSV file "Ratings_Warriner_et_al.csv"

- **arousal** decent job of detecting level of arousal
    able to differentiate between 
        - HIGH arousal "I want my fucking money back" 
        - low arousal "It must have been some little misunderstanding"

- **valence** doesn't seem to be as useful

### Transformer based

_01 Direct VAD regression (Valence–Arousal–Dominance)_

Uses a model trained to output continuous V, A, D scores per text (trained on EmoBank, whose VAD labels are on a 1–5 scale).

```python
# pip install -q transformers torch pandas
import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok_vad = AutoTokenizer.from_pretrained("RobroKools/vad-bert")
mdl_vad = AutoModelForSequenceClassification.from_pretrained("RobroKools/vad-bert")
mdl_vad.eval()

def vad_bert(text: str):
    """Returns dict with valence, arousal, dominance + 0–1 normalized versions."""
    inputs = tok_vad(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = mdl_vad(**inputs).logits.squeeze().tolist()  # [V, A, D]
    v, a, d = out
    # EmoBank uses ~1–5 range; normalize to 0–1 for dashboards
    v01 = (v - 1) / 4
    a01 = (a - 1) / 4
    d01 = (d - 1) / 4
    return {"vad_valence": v, "vad_arousal": a, "vad_dominance": d,
            "vad_valence_01": v01, "vad_arousal_01": a01, "vad_dominance_01": d01}

```

_02 GoEmotions (28-label multi-label emotions → arousal)_

GoEmotions is a 28-label emotion classifier. We take the full probability vector and map labels to arousal weights, then compute an expected arousal (probability-weighted). Tune the weights to your domain.

```python
from transformers import pipeline

goemo = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None  # return all 28 label scores
)

# Simple, editable arousal weights (0=very calm .. 1=very activated)
GO_AROUSAL = {
    "anger": .90, "annoyance": .70, "disgust": .80, "fear": .90, "nervousness": .85,
    "surprise": .80, "excitement": .80, "remorse": .65, "sadness": .30, "grief": .25,
    "disappointment": .40, "disapproval": .55, "confusion": .55, "embarrassment": .60,
    "curiosity": .55, "desire": .60, "joy": .70, "love": .60, "admiration": .55,
    "gratitude": .45, "optimism": .50, "pride": .55, "caring": .45, "relief": .30,
    "realization": .40, "approval": .45, "amusement": .60, "neutral": .20,
}

def arousal_from_goemotions(text: str):
    preds = goemo(text)[0]  # list of {'label', 'score'}
    # probability-weighted expected arousal
    ea = sum(GO_AROUSAL.get(p["label"], 0.5) * p["score"] for p in preds)
    # 'intensity' that emphasises negative high-arousal (optional)
    neg_prob = sum(p["score"] for p in preds if p["label"] in {"anger","annoyance","disgust","fear","sadness","grief","remorse","disappointment","disapproval","nervousness"})
    intensity = ea * neg_prob
    return {"go_arousal": ea, "go_neg_intensity": intensity}

# df = df.join(df["complaint_text"].apply(arousal_from_goemotions).apply(pd.Series))

```

_02 Ekman style 7-class emotions_

A compact, robust model (j-hartmann/emotion-english-distilroberta-base) with labels: anger, disgust, fear, joy, neutral, sadness, surprise. We again map label probs → arousal.

```python
ekman = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

EKMAN_AROUSAL = {  # heuristic weights
    "anger": .90, "disgust": .70, "fear": .90, "joy": .70,
    "sadness": .30, "surprise": .80, "neutral": .20
}

def arousal_from_ekman(text: str):
    preds = ekman(text)[0]
    ea = sum(EKMAN_AROUSAL[p["label"]] * p["score"] for p in preds)
    # optional intensity focusing on negative, high arousal
    neg_prob = sum(p["score"] for p in preds if p["label"] in {"anger","disgust","fear","sadness"})
    return {"ek_arousal": ea, "ek_neg_intensity": ea * neg_prob}

# df = df.join(df["complaint_text"].apply(arousal_from_ekman).apply(pd.Series))

```

putting it all together, apply to case data

```python
# GoEmotions
df = df.join(df["complaint_text"].apply(arousal_from_goemotions).apply(pd.Series))

# Ekman 7-class
df = df.join(df["complaint_text"].apply(arousal_from_ekman).apply(pd.Series))

# Direct VAD regression
df = df.join(df["complaint_text"].apply(vad_bert).apply(pd.Series))

```

## 02 Forecasting
forecast sentiment trends using ARIMA and time series analysis

## 03 Regression
Predict sentiment from other attributes in the data set


