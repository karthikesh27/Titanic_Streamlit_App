import streamlit as st
import pandas as pd
import joblib
model = joblib.load("model.joblib")
FEATURES = ["Pclass","Sex","Age","Fare"]

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.title("Titanic Survival Prediction")

st.write("Predict whether a passenger would survive.")

pclass = st.selectbox("Passenger Class",[1,2,3])

sex = st.radio("Gender",["Female","Male"])

age = st.slider("Age",1,80,30)

fare = st.slider("Fare",0,500,50)

if st.button("Predict"):

    model = load_model()

    sex = 1 if sex=="Female" else 0

    data = pd.DataFrame([[pclass,sex,age,fare]],columns=FEATURES)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if prediction==1:
        st.success("Passenger is likely to Survive")
    else:
        st.error("Passenger is unlikely to Survive")

    st.metric("Survival Probability",f"{probability*100:.2f}%")
