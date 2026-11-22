# -*- coding: utf-8 -*-
"""Executable Python version of Activity_01/rnn_text_classification.ipynb."""


# 1. IMPORT LIBRARIES
import pandas as pd
import torch
import torch.nn as nn
import tiktoken

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



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


# Use 3000 reviews as instructed by sir
df = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)

# Convert sentiment to numbers
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

print("\nDataset shape:", df.shape)
print(df["sentiment"].value_counts())

# 3. TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    df["review"].values,
    df["label"].values,
    test_size=0.20,
    random_state=42,
    stratify=df["label"].values
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. TOKENIZATION USING TIKTOKEN

enc = tiktoken.get_encoding("cl100k_base")

# Maximum review length
MAX_LEN = 150

# Tokenize all reviews
train_tokens = [enc.encode(text) for text in X_train]
test_tokens = [enc.encode(text) for text in X_test]

print("\nExample tokens:")
print(train_tokens[0][:10])

# 5. CREATE COMPACT VOCABULARY

all_tokens = set()

for tokens in train_tokens:
    all_tokens.update(tokens)

for tokens in test_tokens:
    all_tokens.update(tokens)


token_to_id = {
    token: i + 1
    for i, token in enumerate(all_tokens)
}

VOCAB_SIZE = len(token_to_id) + 1
PAD_ID = 0

print("Vocabulary size:", VOCAB_SIZE)

# 6. CONVERT TOKENS + PAD/TRUNCATE


def prepare_sequence(tokens):

    # Convert tiktoken IDs to compact IDs
    sequence = [
        token_to_id.get(token, 0)
        for token in tokens
    ]

    # Truncate
    sequence = sequence[:MAX_LEN]

    # Padding
    if len(sequence) < MAX_LEN:
        sequence += [PAD_ID] * (MAX_LEN - len(sequence))

    return sequence


X_train_processed = [
    prepare_sequence(tokens)
    for tokens in train_tokens
]

X_test_processed = [
    prepare_sequence(tokens)
    for tokens in test_tokens
]

# 7. CONVERT TO PYTORCH TENSORS

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

print("\nTensor shape:", X_train_tensor.shape)

# 8. CREATE DATA LOADER


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# 9. BUILD RNN MODEL

class IMDB_RNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=64
    ):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=PAD_ID
        )

        # Bidirectional RNN
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Output layer
        self.fc = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):

        # Embedding
        x = self.embedding(x)

        # RNN
        output, _ = self.rnn(x)

        # Mean pooling
        mask = (
            x.abs().sum(dim=2) != 0
        ).float()

        mask = mask.unsqueeze(2)

        output = output * mask

        pooled = output.sum(dim=1) / (
            mask.sum(dim=1) + 1e-8
        )

        # Linear output
        return self.fc(pooled).squeeze(1)


# ============================================================
# 10. CREATE MODEL
# ============================================================

model = IMDB_RNN(
    VOCAB_SIZE
).to(device)

print("\nModel created successfully.")


# ============================================================
# 11. LOSS FUNCTION + OPTIMIZER
# ============================================================

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 12. TRAIN RNN
# ============================================================

EPOCHS = 12

print("\n========== TRAINING ==========")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)

        # Calculate loss
        loss = loss_function(
            outputs,
            y_batch
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Statistics
        total_loss += loss.item()

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)

    train_accuracy = correct / total

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Accuracy: {train_accuracy * 100:.2f}%"
    )


# ============================================================
# 13. TEST ACCURACY
# ============================================================

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
    ).float().mean()

print("\n========== FINAL RESULT ==========")
print(
    f"Test Accuracy: {test_accuracy.item() * 100:.2f}%"
)


# ============================================================
# 14. FUNCTION TO TEST YOUR OWN REVIEW
# ============================================================

def predict_sentiment(text):

    # Tokenize using tiktoken
    tokens = enc.encode(text)

    # Prepare sequence
    sequence = prepare_sequence(tokens)

    # Tensor
    x = torch.tensor(
        [sequence],
        dtype=torch.long
    ).to(device)

    # Prediction
    model.eval()

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


# ============================================================
# 15. TEST POSITIVE REVIEW
# ============================================================

predict_sentiment(
    "This movie was amazing and I really enjoyed it"
)


# ============================================================
# 16. TEST NEGATIVE REVIEW
# ============================================================

predict_sentiment(
    "This movie was boring and terrible"
)
