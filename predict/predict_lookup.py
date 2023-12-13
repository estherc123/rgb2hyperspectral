import pandas as pd
import numpy as np

# Load the data
csv_file = '/content/drive/MyDrive/hyperspectral_database/output_vegetation.csv'
df = load_data(csv_file)

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df['reflectances'] = df['reflectances'].apply(lambda x: [float(i) for i in x.split(',')])
    return df

def find_closest_rgb(input_rgb, df):
    # Calculate the Euclidean distance for each RGB set in the DataFrame
    distances = np.sqrt(np.sum((df[['R', 'G', 'B']] - np.array(input_rgb))**2, axis=1))
    closest_index = distances.idxmin()
    return df.iloc[closest_index]


def predict(input_rgb):
    # Find the closest RGB in the dataset
    closest_match = find_closest_rgb(input_rgb, df)

    # Retrieve the corresponding reflectance data
    closest_reflectance = closest_match['reflectances']
    return closest_reflectance
