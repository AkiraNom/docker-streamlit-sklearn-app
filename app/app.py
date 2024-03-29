#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import streamlit as st
import time

import plotly.express as px

from utils import make_sidebar, cover_page, load_dataset, select_ml_algorithm

warnings.filterwarnings('ignore')

st.set_page_config(
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    # initial_sidebar_state="expanded",
)

make_sidebar()
cover_page()

df, target_class_name = load_dataset(st.session_state['dataset'])

st.header('Data', divider='orange')
if 'target' not in st.session_state:
    if 'target' in df.columns.tolist():
        st.session_state['target'] = 'target'
    else:
        st.session_state['target'] = df.columns.tolist()[-1]

col1, col2, col3 = st.columns([0.2,0.5,2])
with col1:
    st.write('')
with col2:
    if st.checkbox('Specify target class'):
        st.write(st.session_state['target'])
        default_ix = df.columns.tolist().index(st.session_state['target'])
        selected_target = st.selectbox('Select target feature', df.columns.tolist(), index=default_ix)
        st.session_state['target'] = selected_target
with col3:
    st.info('If the target class is neither in the target column nor at the last column')
tab1, tab2 = st.tabs(['Data Table','Column Summary'])

with tab1:
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    for col_name in df.columns[:-1]:
        with st.container():
            with st.expander(col_name):
                col1, col2, col3, col4 = st.columns([1,4,3,3])
                with col1:
                    st.write('')
                with col2:
                    if st.checkbox('Show target class',key=col_name):
                        color = st.session_state['target']
                    else:
                        color = None

                    fig = px.histogram(df, x= col_name, color=color,width=500, height=300)
                    st.plotly_chart(fig)
                with col3:
                    st.write('')
                with col4:
                    stats = df[col_name].describe()
                    stats.loc['missing'] = df[col_name].isnull().sum()
                    st.write(stats)

st.header('Data Inspection', divider='orange')

col1, col2 = st.columns(2)
with col1:
    st.subheader('Scatterplot Matrix')
    st.write('Visualization of the relationship between each pair of variables')
    pair_fig =  sns.pairplot(df, hue=st.session_state['target'],
                             markers=['o','s','D'],
                             corner=True)
    st.pyplot(pair_fig)

with col2:
    st.subheader('Correlation plot')
    st.write('compute the coreraltion coefficient')
    corr = df.drop(st.session_state['target'],axis=1).corr()
    corr_fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr,
                annot=True,
                ax=ax,
                )
    st.pyplot(corr_fig)



# ---------- Missing data handling ---------------

if 'test' not in st.session_state:
    st.session_state['test'] = 0
else:
    st.session_state['test'] +=1

st.write(f'''{st.session_state['test']}''')
st.session_state['target'] = 'target'

with st.form('Missing value imputation'):
    st.markdown('<b>1. Missing Data Handling</b>', unsafe_allow_html=True)
    n_nulls = df.isnull().sum().sum()
    if 'impute' not in st.session_state:
        st.session_state['impute'] = False
    if n_nulls != 0:
        with st.expander('Missing Data Handling'):
            cols = st.columns([0.5,2,0.5,4])
            with cols[0]:
                st.write('')
            with cols[1]:
                st.dataframe(df.isnull().sum().to_frame('Number of Nulls'), use_container_width=True)
                missing_val_cols = df.columns[df.isnull().any()].tolist()

            with cols[2]:
                st.write('')
            with cols[3]:
                st.write('Define the strategy to fill the missing values or drop rows')
                impute_strategies = ('mean','most_frequent','median', 'constant', 'drop')
                st.write('')
                for col in missing_val_cols:
                    form_cols = st.columns([1,2,3])
                    with form_cols[0]:
                        st.write('Feature')
                        st.markdown(f'&nbsp;&nbsp;&nbsp;<b>{col} </b>', unsafe_allow_html=True)
                    with form_cols[1]:
                        impute_strategy=st.selectbox('Impute method',
                                                     options=impute_strategies,
                                                     key=col,
                                                     help='Drop: this option drops a row containing a null value \n'
                                                     )
                        st.session_state[f'{col}_impute_strategy'] = impute_strategy
                    with form_cols[2]:
                        fill_missing_constant = st.number_input('Constant to fill missing values',
                                                                key=f'missing_value_constant_input_{col}',
                                                                help='If imputation strategy constant is selected')
                        if impute_strategy != 'constant':
                            st.session_state['impute_fill_constant'] = None
                        else:
                            st.session_state['impute_fill_constant'] = fill_missing_constant
                    st.divider()

    else:
        st.info('There is no missing value')

# ----------- Feature selection -----------------------------------
    if 'features_included' not in st.session_state:
        st.session_state['features_included'] = df.drop(st.session_state['target'], axis=1).columns.tolist()
    if 'features_excluded' not in st.session_state:
        st.session_state['features_excluded'] = None

    st.markdown('<b>2. Feature Selection</b>', unsafe_allow_html=True)
    with st.expander('Feature Selection'):
        include_options = st.multiselect(
        'Features included in the model',
        df.drop(st.session_state['target'], axis=1).columns.tolist(),
        df.drop(st.session_state['target'], axis=1).columns.tolist())

        st.write('The model includes : ', include_options)
        exclude_options = [x for x in df.drop(st.session_state['target'], axis=1).columns.tolist() if x not in include_options]

        st.session_state['features_included'] = include_options
        st.session_state['features_excluded'] = exclude_options
#  ----------- Data Normalization -----------------
    st.markdown('<b>3. Data Normalization </b>', unsafe_allow_html=True)
    if 'normalization' not in st.session_state:
        st.session_state['normalization'] = False
    with st.expander('Data Normalization'):
        numeric_features = df.drop(st.session_state['target'], axis=1).select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.drop(st.session_state['target'], axis=1).select_dtypes(include='object').columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Numeric features')
            select_scaler = st.selectbox('Select normalization method',('Min-Max scaling', 'Z-score (Standardization)','Log scaling', 'None'))
            with st.popover('Explanation: Normalization technique'):
                st.markdown('''
                            Min-Max scaling and Z-socre normalization (standardization) are the two fundamental techniques for normalization\n
                            - <b>Min-Max Scaling</b> - It transforms features within a specific range usually between 0 and 1\n
                            &emsp;&emsp;&emsp;&emsp;$$X_{normalized} = \\frac{X-X_{min}}{X_{max}-X_{min}}$$\n
                            &emsp;&emsp;&emsp;&emsp;$$X_{standardized} = \\frac{X-\mu}{\sigma}$$\n
                            - <b>Z-score normalization (standardization)</b> - It transforms features to have a mean ($$\mu$$) of 0 and a standard deviation ($$\sigma$$) of 1.\n

                            |Min-Max Scaling|Z-score|
                            |-|-|
                            |scale a data between 0 and 1 | transforms a feature to have a $$\mu$$ of 0 and a $$\sigma$$ of 1|
                            |sensitive to outliers|less sensitive to outliers|
                            |suitable for knn, NN|suitable for linear regression, SVM|

                            - <b>Log scaling</b> - It converts data into a logarithmic scale\n
                            &emsp;&emsp;&emsp;&emsp;$$X_{log} = \log(X)$$
                            ''',
                            unsafe_allow_html=True)

            select_features_scaled = st.multiselect('Select feature to be scaled', numeric_features, numeric_features)
            st.session_state['features_scaled'] = select_features_scaled


        with col2:
            st.subheader('Categorical features')

            if categorical_features:

                select_features_hotencoded = st.multiselect('Select features to be hot encoded', categorical_features, categorical_features)
                st.session_state['feature_encoded'] = select_features_hotencoded

            else:
                st.write('')
                st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;No categorical features are found', unsafe_allow_html=True)
                st.session_state['feature_encoded'] = None

        st.info('Currently the same normalization/hot-encoding methods are applied to all selected numeric/categorical features. It might be benefitial to add a feature to change normalization method per feature')


    st.write('summarize the changes')

    submitted = st.form_submit_button('Apply')

    if submitted:
        st.session_state['normalization'] = True
        st.session_state['impute'] = True

st.write(st.session_state)

# -------- model training ------------------
st.header('Model Training', divider='orange')

st.header('Data preprocessing',divider='orange')

X = df.drop(columns=[st.session_state['target']]).loc[:,st.session_state['features_included']]
Y = df[st.session_state['target']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=st.session_state['test_size'], random_state=st.session_state['random_state'])
st.subheader('Data structure')
with st.container():
    col1, col2, col3 =st.columns([0.5,4,1])
    with col1:
        st.write('')
    with col2:
        st.code(f"""
                    Shape of predictor dataset  :   {X.shape} \n
                    Shape of trainng dataset    :   {x_train.shape} \n
                    Shape of testing dataset    :   {x_test.shape} \n
                    Number of class             :   {len(np.unique(Y))} \n
                    Target column name          :   {st.session_state['target']}\n
                    Target class                :   {(*target_class_name,)} \n
                    Features included           :   {len(st.session_state['features_included'])} features, {st.session_state['features_included']}\n
                    Features excluded           :   {len(st.session_state['features_excluded'])} features, {st.session_state['features_excluded']}\n
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

else:
    pass
# ----------- model prediction --------------------
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

# --------- model evaluation ----------------
st.header('Model Evaluation', divider='orange')

# Overall metrics
y_test_prob = model.predict_proba(x_test)
y_train_pred = model.predict(x_train)

with st.container():
    col1, col2, col3 = st.columns([0.5,4,1])
    with col1:
        st.write('')
    with col2:
        st.code(f'''
                Train Accuracy      :   {accuracy_score(y_train, y_train_pred):.5f}\n
                Overall Accuracy    :   {accuracy_score(y_test, y_test_preds):.5f}\n
                Overall Precision   :   {precision_score(y_test, y_test_preds, average='macro'):.5f}\n
                Overall Recall      :   {recall_score(y_test, y_test_preds, average='macro'):.5f}\n
                Average AUC         :   {roc_auc_score(y_test,y_test_prob, multi_class='ovr'):.5f}\n
                ''')
    with col3:
        st.write('')

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



