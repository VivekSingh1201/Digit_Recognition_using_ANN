# ============================================================
#         MNIST Digit Recognition - Complete ANN Program
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# ── 1. Load Dataset ──────────────────────────────────────────
train_path = "datasets/mnist/train.csv"
test_path  = "datasets/mnist/test.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print(" Dataset not found!")
    print("   Place train.csv and test.csv inside: datasets/mnist/")
    exit()

print(" Loading datasets...")
mnist_train = pd.read_csv(train_path)
mnist_test  = pd.read_csv(test_path)

print(f"   Train shape : {mnist_train.shape}")
print(f"   Test shape  : {mnist_test.shape}")

# ── 2. Visualize First Digits ─────────────────────────────────
train_data_digit1  = np.asarray(mnist_train.iloc[0:1, 1:]).reshape(28, 28)
test_data_digit1   = np.asarray(mnist_test.iloc[0:1, ]).reshape(28, 28)
train_label_1      = mnist_train.iloc[0, 0]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(train_data_digit1, cmap=plt.cm.gray_r)
plt.title(f"First digit in train data\n(Label: {train_label_1})", fontsize=13)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(test_data_digit1, cmap=plt.cm.gray_r)
plt.title("First digit in test data\n(Label: Unknown)", fontsize=13)
plt.axis('off')

plt.suptitle("MNIST Dataset - First Sample Visualization", fontsize=15, fontweight='bold')
plt.show(block=False)   # ← doesn't stop the program
plt.pause(7)            # ← shows plot for 7 seconds then continues
plt.close()

# ── 3. Feature Engineering ────────────────────────────────────
print("\n Splitting features and labels...")
X_train = mnist_train.iloc[:, 1:]    # pixel values (784 columns)
Y_train = mnist_train.iloc[:, 0]     # labels (digit 0–9)

print(f"   X_train shape : {X_train.shape}")
print(f"   Y_train shape : {Y_train.shape}")

# ── 4. Build & Train ANN Model ────────────────────────────────
print("\n Building ANN model...")
nn_model = MLPClassifier(hidden_layer_sizes=(50))

print(" Training... (this may take a minute)")
nn_model.fit(X_train, Y_train)
print(" Training complete!")

# ── 5. Predict First Test Digit ───────────────────────────────
prediction = nn_model.predict(mnist_test.iloc[0:1, ])
print(f"\n Predicted digit for first test image: {prediction[0]}")

# ── 6. Check Overall Accuracy ─────────────────────────────────
train_accuracy = nn_model.score(X_train, Y_train) * 100
print(f" Training Accuracy: {train_accuracy:.2f}%")

# report based on tain data
print(classification_report(Y_train,prediction))
