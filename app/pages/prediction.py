from lime import lime_tabular
import numpy as np
import pandas as pd
import streamlit as st

from utils import load_model, make_sidebar, local_css, warning_dataset_load, warning_build_model

from sklearn import set_config
set_config(transform_output = "pandas")

# load local css for sidebar
local_css('./style.css')
make_sidebar()

warning_dataset_load()
warning_build_model()

target_class_names = st.session_state['data']['target_class_names']
features = st.session_state['model']['features']['included']
df = st.session_state['object']['dataframe']
x_train = st.session_state['object']['x_train']
x_test = st.session_state['object']['x_test']
y_test = st.session_state['object']['y_test']

st.title('Predict a class with a trained model')
st.write('')
st.write('')
st.write('')

# use a saved model
# saved_model = st.checkbox('Use a saved model')
# if saved_model:
#     model_classifier = load_model()

model_classifier = st.session_state['object']['trained_model']
sliders = []
col1, col2 = st.columns(2, gap='large')
with col1:
    for feature in features:
        ing_slider = st.slider(label=feature, min_value=float(df[feature].min()), max_value=float(df[feature].max()))
        sliders.append(ing_slider)

df_prediction_input = pd.DataFrame(sliders, index=features).transpose()

with col2:
    st.write('Input parameters')
    st.dataframe(df_prediction_input, hide_index=True)

    prediction = model_classifier.predict(df_prediction_input)

    st.markdown(f"## Model Prediction : <strong style='color:tomato;'>{prediction[0]}</strong>", unsafe_allow_html=True)

    probs = model_classifier.predict_proba(df_prediction_input)
    probability = probs[0,0]

    st.write('')
    st.write('')

    st.metric(label='Model Confidence', value=f'{probability*100:.2f} %', delta=f'{(probability-0.5)*100:.2f}')

