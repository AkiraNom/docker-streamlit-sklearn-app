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
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

make_sidebar()
cover_page()

df, target_class_name = load_dataset(st.session_state['dataset'])

st.header('Data', divider='orange')

tab1, tab2 = st.tabs(['Data Table','Column Summary'])

with tab1:
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    for col in df.columns[:-1]:
        with st.container():
            with st.expander(f'{col}'):
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


st.header('Data preprocessing',divider='orange')

with st.expander('Missing Data Handling'):
    col1, col2 = st.columns([0.5,4])
    with col1:
        st.write('')
    with col2:
        st.write(df.isnull().sum())

with st.expander('Data Normalization'):
    st.write('----------')

st.header('Data Inspection', divider='orange')

col1, col2 = st.columns(2)
with col1:
    st.subheader('Scatterplot Matrix')
    st.write('Visualization of the relationship between each pair of variables')
    pair_fig =  sns.pairplot(df, hue='target',
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

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=st.session_state['test_size'], random_state=st.session_state['random_state'])

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
                    Target Class                :   {(*target_class_name,)} \n
                    Random_state                :   {st.session_state['random_state']}
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

y_test_preds = model.predict(x_test)

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
    st.subheader('Feature Importance')

    if st.session_state['algorithm']!='Support Vector Machines':

        st.write('')
        if st.session_state['algorithm'] != 'Logistic Regression':
            coefficients = model.feature_importances_
        else:
            coefficients = model.coef_

        avg_importance = np.mean(np.abs(coefficients), axis=0)
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=True)

        st.plotly_chart(px.bar(feature_importance, x='Importance', y="Feature", orientation='h'), use_container_width=True)
    else:
        st.write('')
        st.write(f'''{st.session_state['algorithm']} don't offer a direct feature importance calculation''')


st.header('Model Evaluation', divider='orange')
st.subheader("Classification Report")
with st.expander('Details of classification report'):
    with st.container():
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f'''
                    - <b>Accuracy</b> - It represents the ratio of correctly predicted labels to the total predicted labels\n
                        &emsp;&emsp;&emsp;&emsp;$$Accuracy = (TP + TN) / (TP + FP + TN + FN)$$
                    - <b>Precision (Positive predictive rate)</b>- It represents the ratio of number of actual positive class correctly predicted to
                    the total number of predicted positive class\n
                        &emsp;&emsp;&emsp;&emsp;$$  Precision = TP / (TP+FP)  $$
                    - <b>Recall (Sensitivity)</b> - It represents the ratio of number of actual positive class correctly predicted
                    to the total number of actual positive class \n
                        &emsp;&emsp;&emsp;&emsp;$$Recall = TP / (TP+FN)$$
                    - <b>f1-score</b> - It is a weighted harmonic mean of precision and recall normalized between 0 and 1.
                    F score of 1 indicates a perfect balance as precision and the recall are inversely related.
                    A high F1 score is useful where both high recall and precision is important. \n
                        &emsp;&emsp;&emsp;&emsp;$$F1-Score = 2 (Precision*recall) / (Precision + recall)$$
                    - <b>support</b> - It represents the number of actual occurrences of the class in the test data set.
                    Imbalanced support in the training data may indicate the need for stratified sampling or rebalancing \n
                    ''',
                    unsafe_allow_html=True)

        with col2:
            st.write('')
            st.markdown('<b>Confusion Matrix </b>', unsafe_allow_html=True)
            st.write('')
            st.image('https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg')
st.write('')
with st.container():
    col1, col2, col3 = st.columns([1,3,1])
    with col1:
        st.write('')
    with col2:
        st.markdown('<b>Summary statics</b>', unsafe_allow_html=True)
        # selected_precision = st.slider('Precision', min_value=1, max_value=10, value=5, step=1)
        df_classification_report = pd.DataFrame.from_dict(classification_report(y_test,
                                                                  y_test_preds,
                                                                  target_names=target_class_name,
                                                                  output_dict=True))\
                                                                      .transpose()\
                                                                      .style.format(precision=5)
        st.dataframe(df_classification_report, use_container_width=True)
    with col3:
        st.write('')


# -------------------------------------------------
# prediction with your input
# https://builtin.com/data-science/feature-importance


