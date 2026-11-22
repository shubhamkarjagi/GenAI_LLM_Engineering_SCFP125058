# -*- coding: utf-8 -*-
"""Executable Python version of Activity_02/Text Classification with LSTM.ipynb."""

# ============================================================
# ACTIVITY 2 - IMDB TEXT CLASSIFICATION USING LSTM
# COMPLETE CODE - RUN THIS ONE CELL
# ============================================================

import pandas as pd
import torch
import torch.nn as nn
import tiktoken

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

from pathlib import Path
import os
# Prefer a full user-provided IMDB Dataset.csv, otherwise use the synthetic demo dataset.
candidates = [
    Path("IMDB Dataset.csv"), Path("data/IMDB Dataset.csv"), Path("../data/IMDB Dataset.csv"),
    Path("data/IMDB_sample.csv"), Path("../data/IMDB_sample.csv")
]
DATASET_PATH = next((p for p in candidates if p.is_file()), None)
if DATASET_PATH is None:
    raise FileNotFoundError("IMDB dataset missing. See the README dataset instructions.")
print("Dataset:", DATASET_PATH)
df = pd.read_csv(DATASET_PATH)
required_columns = {"review", "sentiment"}
if not required_columns.issubset(df.columns):
    raise ValueError("Dataset must contain review and sentiment columns")
df = df.dropna(subset=["review", "sentiment"])


# Use 3000 reviews for faster lab training
df = df.sample(
    n=min(3000, len(df)),
    random_state=42
).reset_index(drop=True)

# Convert sentiment to numbers
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

print("\nDataset:", df.shape)
print(df["sentiment"].value_counts())


# ------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["review"].values,
    df["label"].values,
    test_size=0.20,
    random_state=42,
    stratify=df["label"].values
)

print("\nTraining:", len(X_train))
print("Testing :", len(X_test))


# ------------------------------------------------------------
# 4. TIKTOKEN TOKENIZATION
# ------------------------------------------------------------

enc = tiktoken.get_encoding("cl100k_base")

train_tokens = [
    enc.encode(text)
    for text in X_train
]

test_tokens = [
    enc.encode(text)
    for text in X_test
]


# ------------------------------------------------------------
# 5. CREATE VOCABULARY
# ------------------------------------------------------------

all_tokens = set()

for tokens in train_tokens:
    all_tokens.update(tokens)

for tokens in test_tokens:
    all_tokens.update(tokens)

# 0 = padding
token_to_id = {
    token: i + 1
    for i, token in enumerate(all_tokens)
}

PAD_ID = 0
VOCAB_SIZE = len(token_to_id) + 1

print("Vocabulary:", VOCAB_SIZE)


# ------------------------------------------------------------
# 6. PAD / TRUNCATE
# ------------------------------------------------------------

MAX_LEN = 150

def prepare_sequence(tokens):

    sequence = [
        token_to_id[token]
        for token in tokens
        if token in token_to_id
    ]

    # truncate
    sequence = sequence[:MAX_LEN]

    # pad
    if len(sequence) < MAX_LEN:
        sequence += [PAD_ID] * (
            MAX_LEN - len(sequence)
        )

    return sequence


X_train_processed = [
    prepare_sequence(tokens)
    for tokens in train_tokens
]

X_test_processed = [
    prepare_sequence(tokens)
    for tokens in test_tokens
]


# ------------------------------------------------------------
# 7. CONVERT TO TENSORS
# ------------------------------------------------------------

X_train_tensor = torch.tensor(
    X_train_processed,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test_processed,
    dtype=torch.long
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
)


# ------------------------------------------------------------
# 8. DATA LOADER
# ------------------------------------------------------------

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# ------------------------------------------------------------
# 9. LSTM MODEL
# ------------------------------------------------------------

class IMDB_LSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=64
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_ID
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):

        # Embedding
        embedded = self.embedding(x)

        # LSTM
        output, _ = self.lstm(embedded)

        # Ignore padding
        mask = (
            x != PAD_ID
        ).float().unsqueeze(2)

        output = output * mask

        # Mean of actual words only
        summed = output.sum(dim=1)

        count = mask.sum(dim=1).clamp(min=1)

        pooled = summed / count

        # Linear output
        result = self.fc(pooled)

        return result.squeeze(1)


# ------------------------------------------------------------
# 10. CREATE MODEL
# ------------------------------------------------------------

model = IMDB_LSTM(
    VOCAB_SIZE
).to(device)

print("\nLSTM model created.")


# ------------------------------------------------------------
# 11. LOSS + OPTIMIZER
# ------------------------------------------------------------

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ------------------------------------------------------------
# 12. TRAIN
# ------------------------------------------------------------

EPOCHS = int(os.getenv("EPOCHS", "10"))

print("\n========== TRAINING ==========")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward
        outputs = model(X_batch)

        # Loss
        loss = loss_function(
            outputs,
            y_batch
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # Training accuracy
        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

    accuracy = correct / total

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ------------------------------------------------------------
# 13. TEST ACCURACY
# ------------------------------------------------------------

model.eval()

X_test_device = X_test_tensor.to(device)
y_test_device = y_test_tensor.to(device)

with torch.no_grad():

    test_outputs = model(
        X_test_device
    )

    test_predictions = (
        torch.sigmoid(test_outputs) >= 0.5
    ).float()

    test_accuracy = (
        test_predictions == y_test_device
    ).float().mean().item()

print("\n========== FINAL RESULT ==========")

print(
    f"LSTM Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# 14. PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_sentiment(text):

    model.eval()

    # Tokenize
    tokens = enc.encode(text)

    # Convert to IDs
    sequence = [
        token_to_id[token]
        for token in tokens
        if token in token_to_id
    ]

    # Truncate
    sequence = sequence[:MAX_LEN]

    # Pad
    if len(sequence) < MAX_LEN:
        sequence += [PAD_ID] * (
            MAX_LEN - len(sequence)
        )

    # Tensor
    x = torch.tensor(
        [sequence],
        dtype=torch.long
    ).to(device)

    # Prediction
    with torch.no_grad():

        output = model(x)

        probability = torch.sigmoid(
            output
        ).item()

    if probability >= 0.5:
        sentiment = "positive"
    else:
        sentiment = "negative"

    print("\nReview:", text)
    print(f"Probability: {probability:.4f}")
    print("Prediction:", sentiment)


# ------------------------------------------------------------
# 15. TEST REVIEWS
# ------------------------------------------------------------

predict_sentiment(
    "This movie was amazing and I really enjoyed it"
)

predict_sentiment(
    "This movie was boring and terrible"
)

predict_sentiment(
    "This movie was not amazing and I really did not enjoy it"
)
