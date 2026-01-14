
# SAMPLE CODE TO LOAD AND USE MODEL

import pickle
import numpy as np

# Load model
model = pickle.load(open('heatwave_risk_model.pkl', 'rb'))

# Sample prediction
sample_data = np.array([[38, 25, 12, 20, 0]])  # max_temp, min_temp, wind_speed, wind_gusts, precipitation
prediction = model.predict(sample_data)[0]

risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
print(f"Predicted Risk: {risk_labels[prediction]}")
