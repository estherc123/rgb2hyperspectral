import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

#load the data
augmented_data = ...

# Prepare the dataset
X = np.array([item['rgb'] for item in augmented_data])  # Features (RGB values)
#y = np.array([item['reflectances'] for item in augmented_data])  # Labels (spectral reflectance)
lengths = [len(item['reflectances']) for item in augmented_data]
unique_lengths = set(lengths)
print("Unique lengths of reflectance arrays:", unique_lengths)

max_length = max(lengths)  # Find the maximum length

# Pad each array to have the same length
padded_reflectances = [np.pad(item['reflectances'], (0, max_length - len(item['reflectances'])), 'constant') for item in augmented_data]


y = np.array(padded_reflectances, dtype=np.float32)
print("Shape of y after padding:", y.shape)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),  # Input layer for 3 RGB values
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(y.shape[1])  # Output layer with a node for each reflectance value
])


model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=400, validation_split=0.1, batch_size = 512)

# Evaluate the model
model.evaluate(X_test, y_test)

#Save model
model.save('/path/to/save')
