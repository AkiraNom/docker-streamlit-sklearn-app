from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             precision_score,
                             recall_score,
                             roc_auc_score)
import scikitplot as skplt
import time
import datetime

from utils import (make_sidebar,
                   local_css,
                   warning_dataset_load,
                   insert_nan_values,
                   create_impute_strategy_selector,
                   optimize_hyperparameters,
                   construct_pipeline,
                   )


# load local css for sidebar
local_css('.app/style.css')
make_sidebar()
st.title('Building ML Model', anchor='build-model')
st.write('')

# extract variables from st.session_state
df = st.session_state['object']['dataframe']
target = st.session_state['data']['target']
target_class_names = target_class_names = st.session_state['data']['target_class_names']
presence_nulls = presence_nulls = st.session_state['preprocessing']['nulls']['presence']


st.header('Data Pre-processing', anchor='pre-processing',divider='orange')
# --------- Insert nan values in the dataset -------------------
cols = st.columns([0.1,2,1,1])
with cols[1]:
    st.info('The original dataset does not contain nan values.  \n The button randomly insert np.nan values in the dataset')
with cols[2]:
    st.write('')
    if st.button('Insert NaN'):
        df = insert_nan_values(df,target)


with st.form('Pre-processing'):
# ----------- Feature selection -----------------------------------
    st.markdown('<b>1. Feature Selection</b>', unsafe_allow_html=True)
    with st.expander('Feature Selection'):
        include_options = st.multiselect(
        'Features included in the model',
        df.drop(target, axis=1).columns.tolist(),
        df.drop(target, axis=1).columns.tolist()[:10])

        exclude_options = [x for x in df.drop(target, axis=1).columns.tolist() if x not in include_options]

# -------------- Missing data handling  -------------------------------------
    st.markdown('<b>2. Missing Data Handling</b>', unsafe_allow_html=True)

    categorical_features = st.session_state['data']['cat_features']
    numeric_features = st.session_state['data']['num_features']

    with st.expander('Missing Data Handling'):
        cols = st.columns([0.2,2,0.5,4])
        with cols[0]:
            st.write('')
        with cols[1]:
            st.dataframe(df.isnull().sum().to_frame('Number of Nulls'), use_container_width=True)
            missing_val_cols = df.columns[df.isnull().any()].tolist()
        with cols[2]:
            st.write('')
        with cols[3]:

            st.write('Define the strategy to fill the missing values')
            st.code('''
                    For categorical features,
                        strategy    :   'most_frequent' (default)
                    For numerical features,
                        strategy    :   'mean' (default) or 'median'
                    ''')

            st.write('')

            # categorical
            st.write('Categorical Features :')
            cat_feature_missing = [feature for feature in missing_val_cols if feature in categorical_features]
            impute_strategy_cat = create_impute_strategy_selector('cat')

            st.divider()

            # numerical
            st.write('Numerical Features :')
            num_feature_missing = [feature for feature in missing_val_cols if feature in numeric_features]
            selected_impute_strategy_num = create_impute_strategy_selector('num')

#  ----------- Data Normalization -----------------
    st.markdown('<b>3. Data Normalization </b>', unsafe_allow_html=True)

    with st.expander('Data Normalization'):

        perform_scaling = False

        if st.checkbox('Perform data normalization'):
            perform_scaling = True


        cols = st.columns([0.1,2])
        with cols[1]:
            st.subheader('Numeric features')
            selected_scaler = st.selectbox('Select normalization method',('Min-Max scaling', 'Z-score normalization (Standardization)'))

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

                            ''',
                            unsafe_allow_html=True)

            select_features_scaled = st.multiselect('Select feature to be scaled', numeric_features, numeric_features)

    st.markdown('<b>4. Categorical data transformation </b>', unsafe_allow_html=True)
    with st.expander('Categorical data transformation'):
        cols = st.columns([0.1,2])
        with cols[1]:
            # if other encoding strategy needs to be added,
            # encoding_strategies = ['']
            # encoder = st.multiselect('Select feature to be scaled', encoding_strategies, encoding_strategies)
            encoder = st.session_state['preprocessing']['encoding']['encoder']
            st.write('Categorical data transformation')
            st.code(f'Encoder : {encoder}')

            st.info('the OneHotEncoder finds the unique values per feature and transform the data to a binary one-hot encoding.')

    st.warning('WARNING: Currently the same normalization/encoding are applied to all numeric/categorical features.')
    submitted = st.form_submit_button('Apply')

    if submitted:

        # update session_state based on the form submitted
        # feature selection
        st.session_state['model']['features']['included'] = include_options
        if exclude_options is not []:
            st.session_state['model']['features']['excluded'] = exclude_options
        # imputation
        st.session_state['preprocessing']['nulls']['strategy']['num_features'] = selected_impute_strategy_num
        st.session_state['preprocessing']['nulls']['strategy']['cat_features'] = impute_strategy_cat

        # normalization
        st.session_state['preprocessing']['normalization']['scaling'] = perform_scaling
        if perform_scaling:
            st.session_state['preprocessing']['normalization']['features'] = select_features_scaled
            st.session_state['preprocessing']['normalization']['scaler'] = selected_scaler
        # encoding
        # st.session_state['preprocessing']['encoding']['encoder'] = encoder

        with st.container():
            cols =st.columns([0.5,4,1])
            with cols[0]:
                st.write('')
            with cols[1]:

                features_included = st.session_state['model']['features']['included']
                if st.session_state['model']['features']['excluded'] is None:
                    features_excluded = None
                else:
                    features_excluded = (f'''
                                        {len(st.session_state['model']['features']['excluded'])} features,  {st.session_state['model']['features']['excluded']}
                                        ''')

                impute_strategy_num = st.session_state['preprocessing']['nulls']['strategy']['num_features']
                impute_strategy_cat = st.session_state['preprocessing']['nulls']['strategy']['cat_features']

                apply_scaler = st.session_state['preprocessing']['normalization']['scaling']
                scaler = st.session_state['preprocessing']['normalization']['scaler']

                st.code(f"""
                            Shape of predictor dataset  :   {df.drop(columns=target).shape} \n
                            Target column name          :   {target}\n
                            Number of classes           :   {len(np.unique(df[target]))} \n
                            Target class                :   {(*target_class_names,)} \n
                            Features included           :   {len(features_included)} features, {features_included}\n
                            Features excluded           :   {features_excluded}\n
                            Missing values              :   {presence_nulls}\n
                            Imputation strategy         :   numerical features    {impute_strategy_num}
                                                            categorical features  {impute_strategy_cat}\n
                            Normalization               :   {apply_scaler}     {scaler if apply_scaler else ''}\n
                        """)
            with cols[2]:
                st.write('')

# -------- model training ------------------
st.header('Model Training', anchor='model-training',divider='orange')

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
st.header('Model Selection', anchor='model-selection',divider='orange')

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

st.header('Build a model', anchor='build-model', divider='orange')
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
    st.session_state['object']['trained_model'] = model

# ----------- model evaluation --------------------
st.header('Model Evaluation', anchor='model-evaluation',divider='orange')

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
st.subheader('Overall model perfomance', anchor='model-performance')

# Overall metrics
y_test_prob = model.predict_proba(x_test)
y_train_pred = model.predict(x_train)

if st.session_state['data']['dataset'] != 'Breast Cancer':
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
    )

# save model
model_path = './data/ml_model.model'
dump(model, model_path)




