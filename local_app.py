import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import *

path = "C:/Users/MCS/OneDrive - Universidad Santo Tomás/Inteligencia Artificial/Codigos Propios/Actividad Individual/"
dataset = pd.read_csv(path + "dataset_APP.csv",header = 0,sep=";",decimal=",") 
covariables = ['Avg. Session Length', 'Time on App', 'Time on Website',
               'Length of Membership', 'dominio', 'Tec']

with open(path + 'best_model.pkl', 'rb') as model_file:
    dt2 = pickle.load(model_file)

st.title("Predicción del precio basado en el uso del usuario")

dom = st.selectbox("Seleccione el dominio:", ['gmail', 'Otro', 'hotmail', 'yahoo']) 
tec = st.selectbox("Seleccione el dispositivo:", ['Smartphone', 'Portatil', 'PC', 'Iphone']) 

avg = st.text_input("Ingrese Avg. Session Length:", value=32.063775)
time_app = st.text_input("Ingrese Time on App:", value=10.719150)
time_web = st.text_input("Ingrese Time on Website:", value=37.712509)
lenght = st.text_input("Ingrese Length of Membership:", value=3.004743)

if st.button("Calcular"):
    try:
        usuario = pd.DataFrame({
            "x0": [float(avg)],
            "x1": [float(time_app)],
            "x2": [float(time_web)],
            "x3": [float(lenght)],
            "x4": [dom],
            "x5": [tec]})
        base = dataset.get(covariables)
        usuario.columns = base.columns
        base = pd.concat([usuario, base], axis = 0, ignore_index=True)
        prediccion = predict_model(dt2, data=base)
        prediccion = prediccion['prediction_label'].head(1).values[0]
        st.markdown(f"<p class='big-font'>Predicción: {prediccion}</p>", unsafe_allow_html=True)
    except ValueError:
        st.error("Por favor, ingrese valores numéricos válidos en todos los campos.")

if st.button("Reiniciar"):
    st.experimental_rerun()