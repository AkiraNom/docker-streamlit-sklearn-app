#import libraries
import warnings
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

import utils

warnings.filterwarnings('ignore')

# utils.make_sidebar()
utils.cover_page()

st.write('')
st.write('')
st.divider()
st.write('')
st.header('Select Data', anchor='select-data',divider='orange')
with st.form('Datase_main_selector'):

    selected_dataset = st.selectbox('Select Dataset',('Iris','Wine Dataset','Breast Cancer'))

    cols = st.columns([0.4,1])
    with cols[1]:
        if st.form_submit_button('Load Data'):
            utils.clear_cache()
            utils.clear_session_state()
            utils.initialize_session_state()
            df, target_class_names = utils.load_dataset(selected_dataset)
            st.session_state['data'] = {
                                        'dataset' : selected_dataset,
                                        'target' : utils.default_target_class_col(df),
                                        'target_class_names' : target_class_names,
                                        'features' : df.drop(st.session_state['data']['target'], axis=1).columns.tolist()
                                        }
            st.session_state['object']['dataframe'] = df
            st.session_state['model']['features']['included'] = df.drop(st.session_state['data']['target'], axis=1).columns.tolist()
            utils.check_feature_dtype(df, st.session_state['data']['target'])
            utils.check_nulls(df)

utils.warning_dataset_load()

if 'dataset' in st.session_state['data']:
    st.success('Data loaded successfully')

st.divider()
st.write('')
st.subheader('')
st.info('Click a button to perform exploratory data analysis or to build a ML model')
cols = st.columns([0.1,1,1])
with cols[1]:
    switch_page_eda = st.button('Perform EDA')
with cols[2]:
    switch_page_build = st.button('Build a ML model')

if switch_page_eda:
    switch_page('exploratory_data_analysis')
if switch_page_build:
    switch_page('model_training')
