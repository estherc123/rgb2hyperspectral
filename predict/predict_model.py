import tensorflow as tf

def rgb2spectral(input_rgb):
    loaded_model = tf.keras.models.load_model('path/to/model.keras')
    input_rgb = input_rgb.reshape(1, -1)
    # Make the prediction
    predicted_reflectance = loaded_model.predict(input_rgb)
    return predicted_reflectance
