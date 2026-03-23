import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------- Load Data ----------
data = pd.read_csv("student_data.csv")

X = data[['study','sleep','attendance']].values
y = data['pass'].values

# ---------- Train Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ---------- Scaling ----------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------- Model ----------
model = Sequential([
    Dense(8, activation='relu', input_shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------- Training ----------
model.fit(X_train, y_train, epochs=300, verbose=0)

# ---------- UI ----------
st.title("🎓 Student Pass Prediction AI")

study = st.slider("Study Hours", 0.0, 10.0, 5.0)
sleep = st.slider("Sleep Hours", 0.0, 10.0, 6.0)
attendance = st.slider("Attendance %", 0.0, 100.0, 70.0)

if st.button("Predict Result"):
    
    user = sc.transform([[study, sleep, attendance]])
    prob = model.predict(user)[0][0]
    
    st.write("📊 Pass Probability =", round(prob*100,2), "%")
    
    if prob > 0.5:
        st.success("✅ Student Likely PASS")
    else:
        st.error("❌ Student Likely FAIL")