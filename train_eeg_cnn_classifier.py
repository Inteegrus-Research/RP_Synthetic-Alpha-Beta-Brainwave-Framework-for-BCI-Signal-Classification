import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Parameters
SAMPLES = 200
CHANNELS = 8
TIMEPOINTS = 7680
BATCH_SIZE = 8
EPOCHS = 10
DATA_DIR = 'F:\Management\Project Works\SB Synthetic Alpha-Beta Brainwave Framework for BCI Signal Classification/mnt/data/eeg_synthetic_dataset'
MODEL_PATH = 'saved_models/eeg_cnn_model.h5'

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 1) Load & preprocess
def load_and_normalize(fpath):
    df = pd.read_csv(fpath, index_col=0)
    data = df.values.T  # (channels, time)
    return (data - data.mean(axis=1, keepdims=True)) / \
           (data.std(axis=1, keepdims=True) + 1e-6)

file_list = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
X = np.array([load_and_normalize(f) for f in file_list])                  # (200, 8, 7680)
y = np.array([0 if 'alpha' in f else 1 for f in file_list])               # (200,)
y_cat = to_categorical(y, num_classes=2)

# Transpose for Conv1D: (samples, time, channels)
X = X.transpose(0, 2, 1)

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, shuffle=True
)

# 3) Load or build model
if os.path.exists(MODEL_PATH):
    print(f"✅ Found existing model at {MODEL_PATH}. Loading...")
    model = load_model(MODEL_PATH)
else:
    print("🚀 No saved model found. Building and training a new one...")
    model = Sequential([
        Conv1D(16, kernel_size=64, strides=16, activation='relu', input_shape=(TIMEPOINTS, CHANNELS)),
        MaxPooling1D(pool_size=4),
        Conv1D(32, kernel_size=32, strides=8, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # 4) Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    # 6) Plot metrics
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    plt.show()

# 5) Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n📊 Test Loss: {loss:.4f} — Test Accuracy: {acc:.4f}")
