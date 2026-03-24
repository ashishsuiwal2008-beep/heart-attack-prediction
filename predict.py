import joblib
import numpy as np

model = joblib.load("heart_model.pkl")

new_patient = np.array([[55, 1, 140, 250, 1, 150]])

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("High risk of heart attack ⚠️")
else:
    print("Low risk ❤️")