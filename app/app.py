import streamlit as st

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from lime import lime_tabular


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time

import plotly.express as px

from utils import make_sidebar, cover_page, load_dataset, select_ml_algorithm

warnings.filterwarnings('ignore')

st.set_page_config(
    # page_title="Ex-stream-ly Cool App",
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

make_sidebar()
cover_page()

df, target_class_name = load_dataset(st.session_state['dataset'])

st.header('Data', divider='orange')

tab1, tab2 = st.tabs(['Column Summary','Data Table'])

with tab1:
    for col in df.columns[:-1]:
        with st.container():
            with st.expander(f'{col}'):
            # st.markdown(f'''<b>{col} </b>''', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns([1,4,3,3])
                with col1:
                    st.write('')
                with col2:
                    if st.checkbox('Show target class',key=f'{col}'):
                        color = 'target'
                    else:
                        color = None

                    fig = px.histogram(df, x= col, color=color,width=500, height=300)
                    st.plotly_chart(fig)
                with col3:
                    st.write('')
                with col4:
                    stats = df[col].describe()
                    stats.loc['missing'] = df[col].isnull().sum()
                    st.write(stats)

with tab2:
    st.dataframe(df, use_container_width=True, hide_index=True)

st.header('Data preprocessing',divider='orange')

with st.expander('Missing Data Handling'):
    st.write(df.isnull().sum())

with st.expander('Data Normalization'):
    st.write('----------')

st.header('Data Inspection', divider='orange')

col1, col2 = st.columns(2)
with col1:
    st.subheader('Scatterplot Matrix')
    st.write('Visualization of the relationship between each pair of variables')
    pair_fig =  sns.pairplot(df, hue='target',
                            #  palette='husl',
                            markers=['o','s','D'],
                            corner=True)
    st.pyplot(pair_fig)

with col2:
    st.subheader('Correlation plot')
    st.write('compute the coreraltion coefficient')
    corr = df.drop('target',axis=1).corr()
    corr_fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr,
                annot=True,
                ax=ax,
                )
    st.pyplot(corr_fig)

st.header('Model Training', divider='orange')

X = df.drop(columns=['target'])
Y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=st.session_state['test_size'])

st.subheader('Data structure')
with st.container():
    col1, col2, col3 =st.columns([0.5,4,1])
    with col1:
        st.write('')
    with col2:
        st.code(f"""
                    Shape of Predictor dataset  :   {X.shape} \n
                    Shape of Trainng dataset    :   {x_train.shape} \n
                    Shape of Testing dataset    :   {x_test.shape} \n
                    Number of Class             :   {len(np.unique(Y))} \n
                    Target Class                :   {target_class_name} \n
                """)
    with col3:
        st.write('')

st.subheader('Model construction')
start_time = time.time()
model = select_ml_algorithm(st.session_state['algorithm'], st.session_state['params'])
model.fit(x_train, y_train)
elapsed_time = time.time() - start_time

if model:
    col1, col2, col3 = st.columns([0.5,4,1])
    with col1:
        st.write('')
    with col2:
        with st.container():
            st.code(f'''
                    Model algorithm    : {st.session_state['algorithm']} \n
                    Model parameters   : {st.session_state['params']}
                    ''')
            st.success('Model created successfully!', icon="✅")
            st.write('')
            st.code(f'Elapsed time for Model Training: {elapsed_time:.5f} seconds')
    with col3:
        st.write('')

st.header('Model Prediction', divider='orange')
st.divider()

y_test_preds = model.predict(x_test)


st.header("Confusion Matrix | Feature Importances", divider='orange')
col1, col2, col3 = st.columns([1.5,0.5,2])
with col1:
    st.subheader('Confusion Matrix')
    st.write('')
    conf_mat_fig = plt.figure(figsize=(10,10))
    ax1 = conf_mat_fig.add_subplot(111)
    skplt.metrics.plot_confusion_matrix(y_test, y_test_preds, ax=ax1, normalize=True)
    st.pyplot(conf_mat_fig)

with col2:
    st.write('')

with col3:
    if st.session_state['algorithm']=='GBC':
        st.subheader('Feature Importance')
        st.write('')
        coefficients = model.coef_

        avg_importance = np.mean(np.abs(coefficients), axis=0)
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=True)

        feature_importance_plot, ax = plt.subplots(figsize=(6, 4))
        ax.barh(feature_importance['Feature'], feature_importance['Importance'])
        ax.set_xlabel('Importance')
        # ax.set_ylabel('Feature')
        ax.set_title('Feature Importance')

        st.pyplot(feature_importance_plot)


st.header('Model Evaluation', divider='orange')
st.subheader("Classification Report")
st.code(classification_report(y_test, y_test_preds))
