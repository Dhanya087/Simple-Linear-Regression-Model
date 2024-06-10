# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv(r'C:\Users\LENOVO\Downloads\SD.csv')

# Preprocessing
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
X1 = imputer.fit_transform(X)
y1 = imputer.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Streamlit app
st.title('Salary Prediction')

# Input years of experience
experience = st.text_input('Enter years of experience', '')

# Predict button
predict_button = st.button('Predict')

if predict_button:
    exp = float(experience)
    pred_value = regressor.predict([[exp]])
    st.write(f'Predicted Salary for {exp} years of experience:', pred_value)

# Plotting
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary Vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
st.pyplot(plt)

plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary Vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
st.pyplot(plt)





