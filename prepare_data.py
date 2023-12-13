import pandas as pd
import json
import pandas as pd
import os
import re
import numpy as np
import random
# Directory containing your .jsonl files
directory = '/content/drive/MyDrive/hyperspectral_database/ChapterV_Vegetation'

all_data = {}
keywords = ["vegetation", "soil", "rock", "snow", "water"]


start_wavelength1 = 200.0
end_wavelength1 = 2000.0

start_wavelength2 = 200.0
end_wavelength2 = 5000.0
keyword = "soil"

standard_length = 2000
num_large = 0

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    data_start_line = find_data_start(file_path)
    if data_start_line is not None:
        # Read the reflectance data
        reflectance_data = pd.read_csv(file_path, sep='\t', skiprows=data_start_line, header=None)
        reflectance = reflectance_data.iloc[:, 0].values   # Convert to decimal
        if len(reflectance) < 3000:
          end_wavelength = end_wavelength1
          start_wavelength = start_wavelength1
        else:
          end_wavelength = end_wavelength2
          start_wavelength = start_wavelength2
         # print(filename)
        if len(reflectance) < standard_length:
          continue

        reflectance = np.clip(reflectance, 0, None)

        # Generate the wavelength values based on the interval
        interval = (end_wavelength - start_wavelength)/(len(reflectance)-1)
        wavelengths = np.arange(start_wavelength, end_wavelength + interval, interval)

        if len(wavelengths) == len(reflectance):
            if keyword not in all_data:
                all_data[keyword] = []
            all_data[keyword].append({'wavelengths': wavelengths, 'reflectance': reflectance})




def xyz_to_rgb(X, Y, Z):
    # Matrix to convert XYZ to linear RGB, assuming sRGB color space
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])

    # Normalize XYZ values based on D65 white point
    X /= 95.047
    Y /= 100.0
    Z /= 108.883


    rgb_linear = np.dot(M, [X, Y, Z])

    # Apply gamma correction
    rgb = np.where(rgb_linear > 0.0031308,
                   1.055 * (rgb_linear ** (1 / 2.4)) - 0.055,
                   12.92 * rgb_linear)
    return rgb

# Load the D65 illuminant data from the CSV file
d65_data = pd.read_csv('/content/drive/MyDrive/hyperspectral_database/CIE_std_illum_D65.csv', header=None)
d65_wavelengths = d65_data.iloc[:, 0].values  # Assuming the first column is wavelengths
d65_values = d65_data.iloc[:, 1].values       # Assuming the second column is illuminance values

cmf_data = pd.read_csv('/content/drive/MyDrive/hyperspectral_database/lin2012xyz2e_fine_7sf.csv', header=None)
cmf_wavelengths = cmf_data.iloc[:, 0].values  # Wavelengths
x_cmf = cmf_data.iloc[:, 1].values  # x-bar values
y_cmf = cmf_data.iloc[:, 2].values  # y-bar values
z_cmf = cmf_data.iloc[:, 3].values  # z-bar values

def spectral2rgb(wavelengths, reflectances):
  mask = (wavelengths >= 380) & (wavelengths <= 700)
  # Apply the mask to select only the wavelengths and reflectances in the range [380, 7
  cropped_wavelengths = wavelengths[mask]
  cropped_reflectances = reflectances[mask]
  #print(wavelengths)
  # Interpolate the D65 SPD to match your data wavelengths
  i_interp = np.interp(cropped_wavelengths, d65_wavelengths, d65_values)
  #print("i interp: ", i_interp)
  length = len(wavelengths)

  x_bar = np.interp(cropped_wavelengths, cmf_wavelengths, x_cmf)
  y_bar = np.interp(cropped_wavelengths, cmf_wavelengths, y_cmf)
  z_bar = np.interp(cropped_wavelengths, cmf_wavelengths, z_cmf)


  #print("x bar: ", x_bar)
  # print("y bar: ",y_bar)
  # print("z bar: ",z_bar)
  # Calculate XYZ values
  #print("ref: ", reflectances)
  X = np.trapz(cropped_reflectances * i_interp * x_bar, cropped_wavelengths)
  Y = np.trapz(cropped_reflectances * i_interp * y_bar, cropped_wavelengths)
  Z = np.trapz(cropped_reflectances * i_interp * z_bar, cropped_wavelengths)

  # Normalize Y
  Y_norm = np.trapz(i_interp * y_bar, cropped_wavelengths)
  K = 100 / Y_norm  # Normalization constant
  #rgb_value = xyz_to_rgb(K * X, K * Y, K * Z
  #xyz = XYZColor(X/100, Y/100, Z/100)
  #xyz = XYZColor(X/100,Y/100, Z/100)
  #print(xyz)
  rgb_value = xyz_to_rgb(K*X, K*Y, K*Z)
  rgb_value[1] = rgb_value[1] * 1.5  # Enhance the green component; adjust the factor as needed
  rgb_value = np.clip(rgb_value, 0, 1)
  return rgb_value


#conversion to XYZ
import numpy as np
import pandas as pd
import json

results = []
visualizer = []
# Loop through the data
for key in all_data:
    temp = all_data[key]
    for entry in temp:
        wavelengths = np.array(entry['wavelengths'])
        reflectances = np.array(entry['reflectance'])
        rgb_value = spectral2rgb(wavelengths, reflectances)
        #rgb_value = xyz_to_rgb(K*X, K*Y, K*Z)
        #rgb_value = convert_color(xyz, sRGBColor, illuminance = 'd65')
        #rgb_value = normalize(rgb_value)
        visualizer.append(rgb_value)
        print(rgb_value)
        results.append({'material': keyword, 'rgb': rgb_value, 'reflectances': reflectances, 'wavelengths': wavelengths})
        # Add more logic if needed, e.g., storing or printing XYZ values



#this section creates the augmented dataset with linear combinations of spectra
augmented_data = results[:]
 # Function to create a linear combination of two arrays (reflectance or RGB)
def combine_arrays(arr1, arr2, alpha):
    return alpha * arr1 + (1 - alpha) * arr2

# Number of new samples to create
num_new_samples = 40000


for _ in range(num_new_samples):
    # Randomly select two samples from the dataset
    sample1, sample2 = random.sample(results, 2)
    if len(sample1['reflectances']) != len(sample2['reflectances']):
        continue  # Skip this iteration if the two samples have different numbers of wavelengths
    # Generate a random alpha value between 0.3 and 0.7
    alpha = random.uniform(0.2, 0.8)

    # Create a new reflectance array as a linear combination
    new_reflectance = combine_arrays(sample1['reflectances'], sample2['reflectances'], alpha)

    # Create a new RGB value as a linear combination
    new_rgb = spectral2rgb(sample1['wavelengths'], new_reflectance)

    # Create a new dictionary for the augmented sample
    new_sample = {
        'material': 'AugmentedMaterial',  # Placeholder
        'rgb': tuple(new_rgb),            # New RGB value
        'reflectances': new_reflectance,
        'wavelengths': sample1['wavelengths']
    }

    augmented_data.append(new_sample)

#save results
df = pd.DataFrame(results)

# Split the 'rgb' tuple into separate columns
df[['R', 'G', 'B']] = pd.DataFrame(df['rgb'].tolist(), index=df.index)
df.drop('rgb', axis=1, inplace=True)

# Convert the 'reflectances' list to a string if needed
df['reflectances'] = df['reflectances'].apply(lambda x: ','.join(map(str, x)))

# Save the DataFrame to a CSV file
df.to_csv(f'/path/to/save', index=False)
