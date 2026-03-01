import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Page setup
st.set_page_config(page_title="KNN Weather Classifier")
st.title("KNN Weather Classification")

# Dataset
X = np.array([
    [50, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [20, 75]
])

y = np.array([0, 1, 0, 0, 1, 1])
labels = {0: "Sunny", 1: "Rainy"}

# Sidebar inputs
st.sidebar.header("Input Features")
temp = st.sidebar.slider("Temperature", 10, 60, 26)
hum = st.sidebar.slider("Humidity", 50, 95, 78)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Prediction
new_point = np.array([[temp, hum]])
prediction = model.predict(new_point)[0]

st.write(f"### Predicted Weather: {labels[prediction]}")

# Plot
fig, ax = plt.subplots()

# Plot sunny & rainy points
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Sunny")
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Rainy")

# Plot new input point
ax.scatter(temp, hum, marker="*", s=200, label="New Day")

ax.set_xlabel("Temperature")
ax.set_ylabel("Humidity")
ax.set_title("KNN Weather Classification")
ax.legend()
ax.grid(True)

st.pyplot(fig)
