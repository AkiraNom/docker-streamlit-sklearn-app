import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import time

from utils import make_sidebar, optimize_hyperparameters, select_ml_algorithm

st.set_page_config(
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

make_sidebar()

# -------- model training ------------------
st.header('Model Training', divider='orange')

if st.session_state['dataset'] == None:
    st.write('')
    st.write('')
    st.warning('Please select dataset to load on the sidebar')
    st.stop()

df = st.session_state['dataframe']
target_class_names = st.session_state['target_class_names']

X = df.drop(columns=[st.session_state['target']]).loc[:,st.session_state['features_included']]
Y = df[st.session_state['target']]

st.subheader('Data Split')

if 'test_size' not in st.session_state:
    st.session_state['test_size'] = 0.3
if 'random_state' not in st.session_state:
    st.session_state['random_state'] = None

with st.form('Test_size_form'):
    test_size = st.slider(label='Test Data Size', min_value=0.1, max_value=0.9, value=0.3, step=0.01)
    st.info('Default: 0.3 (70% for training, 30% for testing)')
    submitted_test_size = st.form_submit_button('Apply')

    random_state_help_text = '''
    Control reproducible output across multiple function calls\n
    Check further in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    '''

    if st.checkbox('random_state on', help=random_state_help_text):
        selected_random_state = st.number_input('Type random state', step=1)
        st.session_state['random_state'] = selected_random_state
    else:
        pass

if submitted_test_size:

    st.session_state['test_size']  = test_size
    st.success('Test_size/random state updated successfully!', icon='✅')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=st.session_state['test_size'], random_state=st.session_state['random_state'])
st.success('Data splitted successfully!',icon="✅")
st.code(f'''
        Train, Test data size       :   {int((1-st.session_state['test_size'])*100)}%, {int(st.session_state['test_size']*100)}%\n
        Shape of predictor dataset  :   {X.shape} \n
        Shape of trainng dataset    :   {x_train.shape} \n
        Shape of testing dataset    :   {x_test.shape} \n
        ''')

# ---------- Select model algorithm -------------------
st.header('Model Selection', divider='orange')

cols = st.columns([0.2,1,1])

with cols[0]:
    st.write('')
with cols[1]:
    st.write('ML algorithm')
    selected_algorithm = st.selectbox('Select Algorithm',('Logistic Regression','Random Forests','GBC','Decision Tree','KNN','Support Vector Machines'))
    st.session_state['algorithm'] = selected_algorithm
with cols[2]:
    st.write('Tune model parameters')
    st.session_state['params'] = optimize_hyperparameters(st.session_state['algorithm'])

st.divider()

st.subheader('Data/Model structure')
with st.container():
    col1, col2, col3 =st.columns([0.5,4,1])
    with col1:
        st.write('')
    with col2:
        # feature exluded if statement (if not none, show a result)
        feature_excluded = (f'''{len(st.session_state['features_excluded'])} features, {st.session_state['features_excluded']}'''
                            if st.session_state['features_excluded'] !=[] else 'None')

        st.code(f"""
                    Shape of predictor dataset  :   {X.shape} \n
                    Testing data size           :   {st.session_state['test_size']:.2f}\n
                    Shape of trainng dataset    :   {x_train.shape} \n
                    Shape of testing dataset    :   {x_test.shape} \n
                    Number of class             :   {len(np.unique(Y))} \n
                    Target column name          :   {st.session_state['target']}\n
                    Target class                :   {(*target_class_names,)} \n
                    Features included           :   {len(st.session_state['features_included'])} features, {st.session_state['features_included']}\n
                    Features excluded           :   {feature_excluded} \n
                    Model algorithm             :   {st.session_state['algorithm']} \n
                    Model parameters            :   {st.session_state['params']}\n
                    Random_state                :   {st.session_state['random_state']}\n
                """)
    with col3:
        st.write('')


st.write('Initiate bulding a model')
if st.button('Build model'):
    start_time = time.time()

    model = select_ml_algorithm(st.session_state['algorithm'], st.session_state['params'])
    model.fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    st.success('Model created successfully!', icon="✅")
    st.write('')
    st.code(f'Elapsed time for Model Training: {elapsed_time:.5f} seconds')
else:
    st.stop()
