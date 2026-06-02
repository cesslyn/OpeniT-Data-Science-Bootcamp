# Week 1 Day 3 - TensorFlow Model Lab 

import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Load the Cleaned Training Data
training_df = pd.read_csv("week1-day3-cleaned-training-data.csv") 
print(training_df.head()) 

print(training_df.columns) 

# Split features and target
X = training_df.drop(columns=["churned"]) 
y = training_df["churned"] 

# Covert to NumPy arrays
X_values = X.to_numpy().astype("float32") 
y_values = y.to_numpy().astype("float32")

# Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split( 
X_values, 
y_values, 
test_size=0.25, 
random_state=42, 
stratify=y_values if len(np.unique(y_values)) > 1 else None 
)

print(X_train.shape) 
print(X_test.shape) 
print(y_train.shape) 
print(y_test.shape) 

# Scale the Features
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)\

print("/n (1) Why should the scaler be fit on the training data only? ")
print("Answer: Fitting the scaler on the training data only ensures that the scaling parameters (mean and standard deviation) are derived solely from the training set. This prevents data leakage from the test set, which could lead to overly optimistic performance estimates. If we were to fit the scaler on the entire dataset, including the test set, it would allow information from the test set to influence the scaling, thus compromising the integrity of our model evaluation.")
print("/n (2) What problem might happen if the test data is used during scaling setup? ")
print("Answer: If the test data is used during scaling setup, it can lead to data leakage, where information from the test set influences the training process. This can result in an overfitted model that performs well on the test set but poorly on unseen data. The model may learn patterns specific to the test set rather than generalizable patterns, leading to an inaccurate assessment of the model's true performance.")

# Simple Tensorflow Model
model = tf.keras.Sequential([ 
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)), 
    tf.keras.layers.Dense(16, activation="relu"), 
    tf.keras.layers.Dense(8, activation="relu"), 
    tf.keras.layers.Dense(1, activation="sigmoid") 
]) 

# Complie the model
model.compile( 
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=["accuracy"] 
) 

# Model structure
model.summary()

# Train the model
history = model.fit( 
    X_train_scaled, 
    y_train, 
    validation_split=0.2, 
    epochs=25, 
    batch_size=4, 
verbose=1 
) 

# Training history
history.history.keys() 

# Chart
import matplotlib.pyplot as plt 
plt.plot(history.history["loss"], label="Training Loss") 
plt.plot(history.history["val_loss"], label="Validation Loss") 
plt.xlabel("Epoch") 
plt.ylabel("Loss") 
plt.legend() 
plt.title("Training and Validation Loss") 
plt.tight_layout() 
plt.savefig("week1-day3-training-loss.png") 
print("Saved chart: week1-day3-training-loss.png") 

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0) 
print("Test loss:", test_loss) 
print("Test accuracy:", test_accuracy) 

# Prediction probabilities
predicted_probabilities = model.predict(X_test_scaled) 
predicted_classes = (predicted_probabilities >= 0.5).astype("int32") 
print(predicted_probabilities[:5]) 
print(predicted_classes[:5])

print("\n(1) What does a prediction close to 1 mean? ")
print("Answer: A prediction close to 1 indicates that the model is confident that the input data belongs to the positive class (in this case, likely to churn). It suggests a high probability of the event occurring.")
print("\n(2) What does a prediction close to 0 mean?")
print("Answer: A prediction close to 0 indicates that the model is confident that the input data belongs to the negative class (in this case, likely not to churn). It suggests a low probability of the event occurring.")
print("\n(3) Is accuracy enough to judge the model? Why or why not?")
print("Answer: Accuracy alone may not be enough to judge the model, especially if the dataset is imbalanced (i.e., one class is much more frequent than the other). In such cases, a model could achieve high accuracy by simply predicting the majority class, but it would perform poorly on the minority class. Therefore, it's important to also consider other metrics such as precision, recall, F1-score, and the confusion matrix to get a more comprehensive understanding of the model's performance.")