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
    # initial_sidebar_state="collapsed",
    initial_sidebar_state="expanded",
)

make_sidebar()
cover_page()

if st.session_state['dataset'] == None:
    st.write('')
    st.write('')
    st.warning('Please select dataset to load on the sidebar')
    st.stop()

df, target_class_name = load_dataset(st.session_state['dataset'])

st.header('Data', divider='orange')
if 'target' not in st.session_state:
    if 'target' in df.columns.tolist():
        st.session_state['target'] = 'target'
    else:
        st.session_state['target'] = df.columns.tolist()[-1]

cols = st.columns([0.2,0.5,2,0.5])
with cols[0]:
    st.write('')
with cols[1]:
    st.write(f'''Current target class: {st.session_state['target']}''')
    if st.checkbox('Specify target class'):
        default_ix = df.columns.tolist().index(st.session_state['target'])
        selected_target = st.selectbox('Select target feature', df.columns.tolist(), index=default_ix)
        st.session_state['target'] = selected_target
with cols[2]:
    st.info('''Default: 'target' or the last column name in dataset''')
with cols[3]:
    st.write('')

tabs = st.tabs(['Column Summary', 'Data Table'])

with tabs[0]:
    with st.container():
        with st.expander('Table summary'):
            cols = st.columns([0.5,2,1])
            with cols[0]:
                st.write('')
            with cols[1]:
                table_summary = df.describe()
                table_summary.loc['Nulls'] = df.isnull().sum()
                st.write(table_summary)
            with cols[2]:
                st.write()

    for i, col_name in enumerate(df.columns):
        with st.container():
            with st.expander(col_name):
                cols = st.columns([1,4,3,3])
                with cols[0]:
                    st.write('')
                with cols[1]:
                    if st.checkbox('Show target class',key=i):
                        color = st.session_state['target']
                    else:
                        color = None

                    fig = px.histogram(df, x= col_name, color=color,width=500, height=300)
                    st.plotly_chart(fig)
                with cols[2]:
                    st.write('')
                with cols[3]:
                    stats = df[col_name].describe()
                    stats.loc['Nulls'] = df[col_name].isnull().sum()
                    st.write(stats)
with tabs[1]:
    st.dataframe(df, use_container_width=True, hide_index=True)

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

with st.form('Pre-processing'):
# ----------- Feature selection -----------------------------------
    if 'features_included' not in st.session_state:
        st.session_state['features_included'] = df.drop(st.session_state['target'], axis=1).columns.tolist()
    if 'features_excluded' not in st.session_state:
        st.session_state['features_excluded'] = None

    st.markdown('<b>1. Feature Selection</b>', unsafe_allow_html=True)
    with st.expander('Feature Selection'):
        include_options = st.multiselect(
        'Features included in the model',
        df.drop(st.session_state['target'], axis=1).columns.tolist(),
        df.drop(st.session_state['target'], axis=1).columns.tolist())

        exclude_options = [x for x in df.drop(st.session_state['target'], axis=1).columns.tolist() if x not in include_options]

        st.session_state['features_included'] = include_options
        st.session_state['features_excluded'] = exclude_options

# -------------- Missing data handling  -------------------------------------
    st.markdown('<b>2. Missing Data Handling</b>', unsafe_allow_html=True)

    if 'null_status' not in st.session_state:
        st.session_state['null_status'] = False
    if 'impute' not in st.session_state:
        st.session_state['impute'] = False

    n_nulls = df.isnull().sum().sum()
    if n_nulls != 0:
        st.session_state['null_status'] = True
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
                if st.checkbox('Fill missing values'):
                    st.session_state['impute'] = True
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


#  ----------- Data Normalization -----------------
    st.markdown('<b>3. Data Normalization </b>', unsafe_allow_html=True)

    if 'normalization' not in st.session_state:
        st.session_state['normalization'] = False
    if 'scaler' not in st.session_state:
        st.session_state['scaler'] = None
    if 'features_scaled' not in st.session_state:
        st.session_state['features_scaled'] = None
    if 'features_encoded' not in st.session_state:
        st.session_state['features_encoded'] = None

    with st.expander('Data Normalization'):
        if st.checkbox('Perform data normalization'):
            st.session_state['normalization'] = True
        numeric_features = df.drop(st.session_state['target'], axis=1).select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.drop(st.session_state['target'], axis=1).select_dtypes(include='object').columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Numeric features')
            selected_scaler = st.selectbox('Select normalization method',('Min-Max scaling', 'Z-score (Standardization)','Log scaling', 'None'))
            st.session_state['scaler'] = selected_scaler
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
                st.session_state['features_encoded'] = select_features_hotencoded

            else:
                st.write('')
                st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;No categorical features are found', unsafe_allow_html=True)

        st.info('Currently the same normalization/hot-encoding methods are applied to all selected numeric/categorical features. It might be benefitial to add a feature to change normalization method per feature')


    if st.checkbox('test'):
        st.session_state['null_status'] = True

    submitted = st.form_submit_button('Apply')

    if submitted:

        with st.container():
            cols =st.columns([0.5,4,1])
            with cols[0]:
                st.write('')
            with cols[1]:
                if st.session_state['null_status'] &(st.session_state['impute']!=True):
                 st.warning('Null values in your dataset may impact on performance of your machine learning model')
                st.code(f"""
                            Shape of predictor dataset  :   {df.drop(columns=st.session_state['target']).shape} \n
                            Target column name          :   {st.session_state['target']}\n
                            Number of class             :   {len(np.unique(df[st.session_state['target']]))} \n
                            Target class                :   {(*target_class_name,)} \n
                            Features included           :   {len(st.session_state['features_included'])} features, {st.session_state['features_included']}\n
                            Features excluded           :   {len(st.session_state['features_excluded'])} features, {st.session_state['features_excluded']}\n
                            Missing values              :   {st.session_state['null_status']}\n
                            Imputation                  :   {st.session_state['impute']}\n
                            Normalization               :   {st.session_state['normalization']}     {st.session_state['scaler'] if st.session_state['normalization'] else ''}\n


                        """)
            with cols[2]:
                st.write('')

st.write(st.session_state)

# # -------- model training ------------------
# st.header('Model Training', divider='orange')

# st.header('Data preprocessing',divider='orange')

# X = df.drop(columns=[st.session_state['target']]).loc[:,st.session_state['features_included']]
# Y = df[st.session_state['target']]

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=st.session_state['test_size'], random_state=st.session_state['random_state'])
# st.subheader('Data structure')
# with st.container():
#     col1, col2, col3 =st.columns([0.5,4,1])
#     with col1:
#         st.write('')
#     with col2:
#         st.code(f"""
#                     Shape of predictor dataset  :   {X.shape} \n
#                     Shape of trainng dataset    :   {x_train.shape} \n
#                     Shape of testing dataset    :   {x_test.shape} \n
#                     Number of class             :   {len(np.unique(Y))} \n
#                     Target column name          :   {st.session_state['target']}\n
#                     Target class                :   {(*target_class_name,)} \n
#                     Features included           :   {len(st.session_state['features_included'])} features, {st.session_state['features_included']}\n
#                     Features excluded           :   {len(st.session_state['features_excluded'])} features, {st.session_state['features_excluded']}\n
#                     Random_state                :   {st.session_state['random_state']}
#                 """)
#     with col3:
#         st.write('')


# st.subheader('Model construction')

# start_time = time.time()
# model = select_ml_algorithm(st.session_state['algorithm'], st.session_state['params'])
# model.fit(x_train, y_train)
# elapsed_time = time.time() - start_time

# if model:
#     col1, col2, col3 = st.columns([0.5,4,1])
#     with col1:
#         st.write('')
#     with col2:
#         with st.container():
#             st.code(f'''
#                     Model algorithm    : {st.session_state['algorithm']} \n
#                     Model parameters   : {st.session_state['params']}
#                     ''')
#             st.success('Model created successfully!', icon="✅")
#             st.write('')
#             st.code(f'Elapsed time for Model Training: {elapsed_time:.5f} seconds')
#     with col3:
#         st.write('')

# else:
#     pass
# # ----------- model prediction --------------------
# st.header('Model Prediction', divider='orange')

# y_test_preds = model.predict(x_test)

# col1, col2, col3 = st.columns([1.5,0.5,2])
# with col1:
#     st.subheader('Confusion Matrix')
#     st.write('')
#     conf_mat_fig = plt.figure(figsize=(10,10))
#     ax1 = conf_mat_fig.add_subplot(111)
#     skplt.metrics.plot_confusion_matrix(y_test, y_test_preds, ax=ax1, normalize=True)
#     st.pyplot(conf_mat_fig)

# with col2:
#     st.write('')

# with col3:
#     st.subheader('Feature Importance')

#     if st.session_state['algorithm']!='Support Vector Machines':

#         st.write('')
#         if st.session_state['algorithm'] != 'Logistic Regression':
#             coefficients = model.feature_importances_
#         else:
#             coefficients = model.coef_

#         avg_importance = np.mean(np.abs(coefficients), axis=0)
#         feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
#         feature_importance = feature_importance.sort_values('Importance', ascending=True)

#         st.plotly_chart(px.bar(feature_importance, x='Importance', y="Feature", orientation='h'), use_container_width=True)
#     else:
#         st.write('')
#         st.write(f'''{st.session_state['algorithm']} don't offer a direct feature importance calculation''')

# # --------- model evaluation ----------------
# st.header('Model Evaluation', divider='orange')

# # Overall metrics
# y_test_prob = model.predict_proba(x_test)
# y_train_pred = model.predict(x_train)

# with st.container():
#     col1, col2, col3 = st.columns([0.5,4,1])
#     with col1:
#         st.write('')
#     with col2:
#         st.code(f'''
#                 Train Accuracy      :   {accuracy_score(y_train, y_train_pred):.5f}\n
#                 Overall Accuracy    :   {accuracy_score(y_test, y_test_preds):.5f}\n
#                 Overall Precision   :   {precision_score(y_test, y_test_preds, average='macro'):.5f}\n
#                 Overall Recall      :   {recall_score(y_test, y_test_preds, average='macro'):.5f}\n
#                 Average AUC         :   {roc_auc_score(y_test,y_test_prob, multi_class='ovr'):.5f}\n
#                 ''')
#     with col3:
#         st.write('')

# st.subheader("Classification Report")
# with st.expander('Details of classification report'):
#     with st.container():
#         col1, col2 = st.columns([2,1])
#         with col1:
#             st.markdown('''
#                     - <b>Accuracy</b> - It represents the ratio of correctly predicted labels to the total predicted labels\n
#                         &emsp;&emsp;&emsp;&emsp;$$Accuracy = (TP + TN) / (TP + FP + TN + FN)$$
#                     - <b>Precision (Positive predictive rate)</b>- It represents the ratio of number of actual positive class correctly predicted to
#                     the total number of predicted positive class\n
#                         &emsp;&emsp;&emsp;&emsp;$$  Precision = TP / (TP+FP)  $$
#                     - <b>Recall (Sensitivity)</b> - It represents the ratio of number of actual positive class correctly predicted
#                     to the total number of actual positive class \n
#                         &emsp;&emsp;&emsp;&emsp;$$Recall = TP / (TP+FN)$$
#                     - <b>f1-score</b> - It is a weighted harmonic mean of precision and recall normalized between 0 and 1.
#                     F score of 1 indicates a perfect balance as precision and the recall are inversely related.
#                     A high F1 score is useful where both high recall and precision is important. \n
#                         &emsp;&emsp;&emsp;&emsp;$$F1-Score = 2 (Precision*recall) / (Precision + recall)$$
#                     - <b>support</b> - It represents the number of actual occurrences of the class in the test data set.
#                     Imbalanced support in the training data may indicate the need for stratified sampling or rebalancing \n
#                     ''',
#                     unsafe_allow_html=True)

#         with col2:
#             st.write('')
#             st.markdown('<b>Confusion Matrix </b>', unsafe_allow_html=True)
#             st.write('')
#             st.image('https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg')
# st.write('')
# with st.container():
#     col1, col2, col3 = st.columns([1,3,1])
#     with col1:
#         st.write('')
#     with col2:
#         st.markdown('<b>Summary statics</b>', unsafe_allow_html=True)
#         # selected_precision = st.slider('Precision', min_value=1, max_value=10, value=5, step=1)
#         df_classification_report = pd.DataFrame.from_dict(classification_report(y_test,
#                                                                                 y_test_preds,
#                                                                                 target_names=target_class_name,
#                                                                                 output_dict=True))\
#                                                                                     .transpose()\
#                                                                                     .style.format(precision=5)
#         st.dataframe(df_classification_report, use_container_width=True)
#     with col3:
#         st.write('')


# # -------------------------------------------------
# # prediction with your input
# # https://builtin.com/data-science/feature-importance



