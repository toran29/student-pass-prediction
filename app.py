import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data
X = np.array([[6,7,80],[8,6,90],[4,5,60],[9,8,95],[5,6,70],[7,8,85],[3,4,50],[8,7,92]])
y = np.array([1,1,0,1,0,1,0,1])

model = LogisticRegression()
model.fit(X, y)

st.title("🎓 Student Pass Prediction AI")

study = st.slider("Study Hours", 0.0, 10.0, 5.0)
sleep = st.slider("Sleep Hours", 0.0, 10.0, 6.0)
attendance = st.slider("Attendance %", 0.0, 100.0, 70.0)

if st.button("Predict Result"):
    pred = model.predict_proba([[study, sleep, attendance]])[0][1]
    st.write(f"📊 Pass Probability = {pred*100:.2f} %")
    if pred > 0.5:
        st.success("✅ Student Likely PASS")
    else:
        st.error("❌ Student Likely FAIL")
