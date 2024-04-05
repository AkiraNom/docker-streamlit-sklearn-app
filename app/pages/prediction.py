from lime import lime_tabular
import numpy as np
import pandas as pd
import streamlit as st

from utils import load_model, make_sidebar, warning_dataset_load, warning_build_model

from sklearn import set_config
set_config(transform_output = "pandas")

make_sidebar()

st.write(st.session_state)

warning_dataset_load()
warning_build_model()

model_classifier = load_model()
target_class_names = st.session_state['data']['target_class_names']
features = st.session_state['model']['features']['included']
df = st.session_state['object']['dataframe']
x_train = st.session_state['object']['x_train']
x_test = st.session_state['object']['x_test']
y_test = st.session_state['object']['y_test']



sliders = []
col1, col2 = st.columns(2)
with col1:
    for feature in features:
        ing_slider = st.slider(label=feature, min_value=float(df[feature].min()), max_value=float(df[feature].max()))
        sliders.append(ing_slider)

df_prediction_input = pd.DataFrame(sliders, index=features).transpose()

st.dataframe(df_prediction_input)

with col2:
    col1, col2 = st.columns(2, gap="medium")

    prediction = model_classifier.predict(df_prediction_input)

    with col1:
        st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(prediction[0]), unsafe_allow_html=True)

    probs = model_classifier.predict_proba(df_prediction_input)
    probability = probs[0,0]

    with col2:
        st.metric(label="Model Confidence", value="{:.2f} %".format(probability*100), delta="{:.2f} %".format((probability-0.5)*100))

    # explainer = lime_tabular.LimeTabularExplainer(x_train.to_numpy(), mode="classification", class_names=target_class_names, feature_names=features)
    # explanation = explainer.explain_instance(df_prediction_input, model_classifier.predict, num_features=len(features), top_labels=3)
    # interpretation_fig = explanation.as_pyplot_figure(label=prediction[0])
    # st.pyplot(interpretation_fig, use_container_width=True)

