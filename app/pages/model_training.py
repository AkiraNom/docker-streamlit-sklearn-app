from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import streamlit as st
# from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             precision_score,
                             recall_score,
                             roc_auc_score)
import scikitplot as skplt
import time
import datetime

# # set transform output config
# set_config(transform_output = "pandas")

from utils import (make_sidebar,
                   warning_dataset_load,
                   optimize_hyperparameters,
                   select_ml_algorithm,
                   construct_pipeline,
                   save_classification_report)

st.set_page_config(
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

make_sidebar()
st.title('Building ML Model')
st.write('')

# -------- model training ------------------
st.header('Model Training', divider='orange')

warning_dataset_load()

df = st.session_state['object']['dataframe']
target_class_names = st.session_state['data']['target_class_names']
target = st.session_state['data']['target']
X = df.drop(columns=target).loc[:,st.session_state['model']['features']['included']]
Y = df[target]

st.subheader('Data Split')

with st.form('Test_size_form'):
    selected_test_size = st.slider(label='Test Data Size', min_value=0.1, max_value=0.9, value=0.3, step=0.01)
    st.info('Default: 0.3 (70% for training, 30% for testing)')

    random_state_help_text = '''
    Control reproducible output across multiple function calls\n
    Check further in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
    '''

    cols = st.columns([1,2,2])
    with cols[0]:
        st.write('')
        if st.checkbox('random_state on', help=random_state_help_text):
            use_random_state = True
        else:
            use_random_state = False
    with cols[1]:
        selected_random_state = st.number_input('Type random state', step=1)

    st.write('')
    submitted_test_size = st.form_submit_button('Apply')

if submitted_test_size:

    st.session_state['model']['test_size']  = selected_test_size
    if use_random_state:
        st.session_state['model']['random_state'] = selected_random_state
    st.success('Test_size/random state updated successfully!', icon='✅')

test_size = st.session_state['model']['test_size']
random_state = st.session_state['model']['random_state']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
st.session_state['object']['x_train'] = x_train
st.session_state['object']['x_test'] = x_test
st.session_state['object']['y_test'] = y_test
cols = st.columns([0.1,2])
with cols[1]:
    st.code(f'''
            Train, Test data size       :   {int((1-test_size)*100)}%, {int(test_size*100)}%\n
            Shape of predictor dataset  :   {X.shape} \n
            Shape of trainng dataset    :   {x_train.shape} \n
            Shape of testing dataset    :   {x_test.shape} \n
            Random_state                :   {random_state} \n
            ''')

# ---------- Select model algorithm -------------------
st.header('Model Selection', divider='orange')

cols = st.columns([0.1,1,1])

with cols[0]:
    st.write('')
with cols[1]:
    st.write('ML algorithm')
    selected_algorithm = st.selectbox('Select Algorithm',('Logistic Regression','Random Forests','GBC','Decision Tree','KNN','Support Vector Machines'))
    st.session_state['model']['algorithm'] = selected_algorithm
with cols[2]:
    st.write('Tune model parameters')
    st.session_state['model']['params'] = optimize_hyperparameters(st.session_state['model']['algorithm'])

st.divider()

st.subheader('Data/Model structure')
with st.container():
    cols =st.columns([0.1,2])

    with cols[1]:
        features_included = st.session_state['model']['features']['included']
        if st.session_state['model']['features']['excluded'] is None:
            features_excluded = None
            st.write('feature excluded is none')
        else:
            features_excluded = (f'''
                                {len(st.session_state['model']['features']['excluded'])} features,  {st.session_state['model']['features']['excluded']}
                                ''')

        st.code(f"""
                    Shape of predictor dataset  :   {X.shape} \n
                    Testing data size           :   {test_size:.2f}\n
                    Shape of trainng dataset    :   {x_train.shape} \n
                    Shape of testing dataset    :   {x_test.shape} \n
                    Number of class             :   {len(np.unique(Y))} \n
                    Target column name          :   {target}\n
                    Target class                :   {(*target_class_names,)} \n
                    Features included           :   {len(features_included)} features, {features_included}\n
                    Features excluded           :   {features_excluded} \n
                    Model algorithm             :   {st.session_state['model']['algorithm']} \n
                    Model parameters            :   {st.session_state['model']['params']}\n
                    Random_state                :   {st.session_state['model']['random_state']}\n
                """)

st.markdown(' <b> Initiate bulding a model </b>', unsafe_allow_html=True)
cols_button = st.columns([0.1,1])
with cols_button[1]:
    if st.button('Build model'):
        pass
    else:
        st.stop()

cols =st.columns([0.1,2])
with cols[1]:
    date_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    st.session_state['model']['date_time'] = date_time
    start_time = time.time()
    model = construct_pipeline()
    model.fit(x_train, y_train)
    elapsed_time = time.time() - start_time
    st.success('Model created successfully!', icon="✅")
    st.write('')
    st.code(f'Elapsed time for Model Training: {elapsed_time:.5f} seconds')
    st.session_state['model']['elapsed_time'] = elapsed_time
    st.session_state['model']['trained_model'] = model

# ----------- model evaluation --------------------
st.header('Model Evaluation', divider='orange')

y_test_preds = model.predict(x_test)

cols = st.columns([1.5,0.5,2])
with cols[0]:
    st.subheader('Confusion Matrix')
    st.write('')
    conf_mat_fig = plt.figure(figsize=(10,10))
    ax1 = conf_mat_fig.add_subplot(111)
    skplt.metrics.plot_confusion_matrix(y_test, y_test_preds, ax=ax1, normalize=True)
    st.pyplot(conf_mat_fig)

with cols[2]:
    st.subheader('Feature Importance')

    if st.session_state['model']['algorithm']!='Support Vector Machines':

        st.write('')
        if st.session_state['model']['algorithm'] != 'Logistic Regression':
            coefficients = model.named_steps['classifier'].feature_importances_
        else:
            coefficients = model.named_steps['classifier'].coef_

        avg_importance = np.mean(np.abs(coefficients), axis=0)
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=True)

        st.plotly_chart(px.bar(feature_importance, x='Importance', y="Feature", orientation='h'), use_container_width=True)
    else:
        st.write('')
        st.write(f'''{st.session_state['model']['algorithm']} don't offer a direct feature importance calculation''')

# --------- model performance ----------------
st.subheader('Overall model perfomance')

# Overall metrics
y_test_prob = model.predict_proba(x_test)
y_train_pred = model.predict(x_train)

with st.container():
    cols = st.columns([0.1,1.9,0.1])
    with cols[1]:
        st.code(f'''
                Train Accuracy   :   {accuracy_score(y_train, y_train_pred):.5f}\n
                Test Accuracy    :   {accuracy_score(y_test, y_test_preds):.5f}\n
                Test Precision   :   {precision_score(y_test, y_test_preds, average='macro'):.5f}\n
                Test Recall      :   {recall_score(y_test, y_test_preds, average='macro'):.5f}\n
                Average AUC      :   {roc_auc_score(y_test,y_test_prob, multi_class='ovr'):.5f}\n
                ''')

st.subheader("Classification Report")
with st.expander('Details of classification report'):
    with st.container():
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown('''
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
    cols = st.columns([0.1,1.9,0.1])
    with cols[1]:
        st.markdown('<b>Summary statics</b>', unsafe_allow_html=True)
        # selected_precision = st.slider('Precision', min_value=1, max_value=10, value=5, step=1)
        df_classification_report = pd.DataFrame.from_dict(classification_report(y_test,
                                                                                y_test_preds,
                                                                                target_names=target_class_names,
                                                                                output_dict=True))\
                                                                                    .transpose()

        st.dataframe(df_classification_report.style.format(precision=5), use_container_width=True)




st.subheader('Save model and classification report')
cols = st.columns([0.1,1.5,0.5])
with cols[1]:
    st.info('Click a download button will automatically refresh a page')
    st.download_button(
        "Download Model",
        data=pickle.dumps(model),
        file_name='ml_model.pkl',
        # on_click=save_classification_report(df_classification_report)
    )

# save model
model_path = './data/ml_model.model'
dump(model, model_path)




