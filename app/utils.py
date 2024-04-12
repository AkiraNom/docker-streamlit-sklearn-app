from joblib import load
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import estimator_html_repr
from sklearn.svm import SVC
from sklearn import set_config
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page
import time

@st.cache_data
def load_dataset(selected_dataset):
    if selected_dataset=="Iris":
        data = datasets.load_iris()
    elif selected_dataset=="Breast Cancer":
        data = datasets.load_breast_cancer()
    elif selected_dataset=="Wine Dataset":
        data = datasets.load_wine()

    feature_data = data.data
    columns = data.feature_names
    target = data.target
    target_class_names = data.target_names

    df = pd.DataFrame(feature_data, columns = columns)
    df.loc[:,'target'] = target.reshape(-1,1)

    return df, target_class_names

def select_ml_algorithm(algorithm,params):
    if algorithm=='KNN':
        alg = KNeighborsClassifier(n_neighbors=params['K'])
    elif algorithm=='Logistic Regression':
        alg = LogisticRegression(penalty=params['penalty'])
    elif algorithm=='GBC':
        alg = GradientBoostingClassifier(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'])
    elif algorithm=='Random Forests':
        alg = RandomForestClassifier(n_estimators=params['n_estimators'])
    elif algorithm=='Decision Tree':
        alg = DecisionTreeClassifier(max_depth=params['max_depth'],criterion=params['criterion'])
    elif algorithm=='Support Vector Machines':
        alg = SVC(C=params['C'],kernel=params['kernel'])
    return alg

def optimize_hyperparameters(clf_name):
    params = dict()
    if clf_name=='KNN':
        K = st.slider('K',1,15)
        params['K']=K
    elif clf_name=='Logistic Regression':
        penalty = st.selectbox('penalty',('l2','none'))
        params['penalty']=penalty
    elif clf_name=='GBC':
        learning_rate = st.slider('learning_rate',min_value=0.01,max_value=10.0, value=0.1)
        n_estimators = st.slider('n_estimators',min_value=1,max_value=500, value=100)
        params['learning_rate']=learning_rate
        params['n_estimators']=n_estimators
    elif clf_name=='Random Forests':
        n_estimators = st.slider('n_estimators',min_value=0,max_value=1500,value=100)
        params['n_estimators']=n_estimators
    elif clf_name=='Decision Tree':
        criterion=st.selectbox('criterion',('gini','entropy'))
        max_depth = st.slider('max_depth',min_value=1,max_value=15)
        params['criterion']=criterion
        params['max_depth']=max_depth
    elif clf_name=='Support Vector Machines':
        C = st.slider('C',min_value=1,max_value=15)
        kernel=st.selectbox('kernel',('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'))
        params['C']=C
        params['kernel']=kernel

    return params

def clear_cache():
    st.cache_data.clear()

def clear_session_state():
    '''Clear all session_states'''
st.session_state.clear()

def initialize_session_state():
    ''' initialize session_state'''
    st.session_state['data'] = {
        'dataset' : None,
        'target' : None,
        'target_class_names' : None,
        'features' : None,
        'num_features' : None,
        'cat_features' : None,
        'insert_nulls' : False
        }
    st.session_state['preprocessing'] = {
        'nulls' : {
                'presence' : False,
                'strategy' : {'num_features' : 'mean',
                              'cat_features' : 'most_frequent'}
                },
        'normalization' : {
                'scaling' : False,
                'scaler' : 'Min-Max scaling',
                'features' : '',
                },
        'encoding' : {
                'encoder' : 'OneHotEncoder'
                },
        }
    st.session_state['model'] = {
        'algorithm' : '',
        'params' : {},
        'test_size' : 0.3,
        'random_state' : None,
        'features' : {
            'included' : [],
            'excluded' : None
            },
        'date_time' : '',
        'elapsed_time' : ''
    }
    st.session_state['object'] = {
        'dataframe' : None,
        'x_train' : None,
        'y_train' : None,
        'x_test' : None,
        'y_test' : None,
        'trained_model' : None
    }

def default_target_class_col(df):
    ''' set default target class col

        if 'target' in the column:
            use it
        else:
            take the last column as the target class col

        to change the target class col -> select_target_class_column(df, target)
    '''
    if 'target' in df.columns.tolist():
        st.session_state['data']['target'] = 'target'
        return  'target'
    else:
        st.session_state['data']['target'] = df.columns.tolist()[-1]
        return df.columns.tolist()[-1]

def check_feature_dtype(df, target):

    num_features = df.drop(target, axis=1).select_dtypes(include=np.number).columns.tolist()
    cat_features = df.drop(target, axis=1).select_dtypes(exclude=np.number).columns.tolist()

    st.session_state['data']['cat_features'] = cat_features
    st.session_state['data']['num_features'] = num_features

def local_css(path):
  with open(path) as f:
      css = f.read()

  return st.markdown(f'<style>{css}</style>',
                     unsafe_allow_html=True)

def make_sidebar():
    with st.sidebar:
        st.title('Multilabel Machine Learning Classification App')
        st.write('')

        st.subheader('Load Data')
        with st.form('Dataset'):

            selected_dataset = st.selectbox('Select Dataset',('Iris','Wine Dataset','Breast Cancer'))

            cols = st.columns([0.4,1])
            with cols[1]:
                if st.form_submit_button('Load Data'):
                    clear_cache()
                    clear_session_state()
                    initialize_session_state()
                    df, target_class_names = load_dataset(selected_dataset)
                    st.session_state['data'] = {
                                                'dataset' : selected_dataset,
                                                'target' : default_target_class_col(df),
                                                'target_class_names' : target_class_names,
                                                'features' : df.drop(st.session_state['data']['target'], axis=1).columns.tolist()
                                                }
                    st.session_state['object'] = {'dataframe' : df}
                    st.session_state['model']['features']['included'] = df.drop(st.session_state['data']['target'], axis=1).columns.tolist()
                    check_feature_dtype(df, st.session_state['data']['target'])
                    check_nulls(df)

        st.write('')
        st.subheader('Navigation Menu', divider='orange')
        st.page_link('app.py', label = 'Overview')
        st.markdown('''<ul>
                    <li><a href="#select-data">Data Selection</a></li>
                    </ul>''', unsafe_allow_html=True)
        st.page_link('./pages/exploratory_data_analysis.py', label='Exploratory data analysis')
        st.page_link('./pages/model_training.py', label='Build a ML model')
        st.markdown('''<ul>
                    <li><a href="#pre-processing">Data Pre-processing</a></li>
                    <li><a href="#model-training">Data Training</a></li>
                    <li><a href="#model-evaluation">Data Evaluation</a></li>
                    <li><a href="#build-model">Build Model</a></li>
                    </ul>''', unsafe_allow_html=True)
        st.page_link('./pages/prediction.py', label='Predict a class')

        st.divider()

        st.markdown('<b>Clear all data</b>:', unsafe_allow_html=True)
        with st.container():
            cols = st.columns([0.1,1])
            with cols[0]:
                st.write('')
            with cols[1]:
                submitted_clear = st.button('Clear Data')

        if submitted_clear:
            clear_cache()
            clear_session_state()
            success = st.success('All data cleared')
            time.sleep(1)
            success.empty()

@st.cache_data
def load_dataset_description(selected_dataset):
    if selected_dataset=="Iris":
        data = datasets.load_iris()
    elif selected_dataset=="Wine Dataset":
        data = datasets.load_wine()
    elif selected_dataset=="Breast Cancer":
        data = datasets.load_breast_cancer()

def cover_page():
    st.header('Multilabel Machine Learning Classification Model')
    cols = st.columns([0.2,2,3])
    with cols[0]:
        st.write('')
    with cols[1]:
        st.write('''
                 ML web app is developed with scikit-learn and streamlit.\n
                 Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning.\n
                 Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
                 ''')
        icon_cols = st.columns([0.5,2,0.5,2])
        with icon_cols[0]:
            st.write('')
        with icon_cols[1]:
            st.image('./app/img/scikit-learn.png', width=100)
            st.markdown('''[scikit-learn](https://scikit-learn.org/stable/)''', unsafe_allow_html=True)
        with icon_cols[2]:
            st.write('')
        with icon_cols[3]:
            st.image('./app/img/Streamlit.png', width=100)
            st.markdown('&nbsp;&nbsp;&nbsp;[streamlit](https://streamlit.io/)', unsafe_allow_html=True)

    with cols[2]:
        image_slideshow()

def image_slideshow():
    '''
    The codes originate from https://discuss.streamlit.io/t/automatic-slideshow/38342
    by TomJohn
    '''
    return components.html("""
                            <!DOCTYPE html>
                            <html>
                            <head>
                            <meta name="viewport" content="width=device-width, initial-scale=1">
                            <style>
                            * {box-sizing: border-box;}
                            body {font-family: Verdana, sans-serif;}
                            .mySlides {display: none;}
                            img {vertical-align: middle;}

                            /* Slideshow container */
                            .slideshow-container {
                            max-width: 1000px;
                            position: relative;
                            margin: auto;
                            }

                            /* Number and caption text (1/3 etc) */
                            .numbertext {
                            color: #f2f2f2;
                            font-size: 12px;
                            padding: 8px 12px;
                            position: absolute;
                            top: 0;
                            }

                            .active {
                            background-color: #717171;
                            }

                            /* Fading animation */
                            .fade {
                            animation-name: fade;
                            animation-duration: 1.5s;
                            }

                            @keyframes fade {
                            from {opacity: .4}
                            to {opacity: 1}
                            }

                            /* On smaller screens, decrease text size */
                            @media only screen and (max-width: 300px) {
                            .text {font-size: 11px}
                            }
                            </style>
                            </head>
                            <body>

                            <div class="slideshow-container">

                            <div class="mySlides fade">
                            <div class="numbertext">1 / 3 : Iris Data</div>
                            <img src="https://images.unsplash.com/photo-1590377663350-cfc38d4b6e43?q=80&w=3431&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                            style="width:100%">
                            </div>

                            <div class="mySlides fade">
                            <div class="numbertext">2 / 3 : Wine Data</div>
                            <img src=https://images.unsplash.com/photo-1568213816046-0ee1c42bd559?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fHdpbmV8ZW58MHx8MHx8fDA%3D"
                            style="width:100%">
                            </div>

                            <div class="mySlides fade">
                            <div class="numbertext">3 / 3 : Brest Cancer Data</div>
                            <img src=https://images.unsplash.com/photo-1581595219618-375a1a48d324?q=80&w=3540&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                            style="width:100%">
                            </div>
                            </div>

                            <div style="text-align:center">
                            <span class="dot"></span>
                            <span class="dot"></span>
                            <span class="dot"></span>
                            </div>

                            <script>
                            let slideIndex = 0;
                            showSlides();

                            function showSlides() {
                            let i;
                            let slides = document.getElementsByClassName("mySlides");
                            let dots = document.getElementsByClassName("dot");
                            for (i = 0; i < slides.length; i++) {
                                slides[i].style.display = "none";
                            }
                            slideIndex++;
                            if (slideIndex > slides.length) {slideIndex = 1}
                            for (i = 0; i < dots.length; i++) {
                                dots[i].className = dots[i].className.replace(" active", "");
                            }
                            slides[slideIndex-1].style.display = "block";
                            dots[slideIndex-1].className += " active";
                            setTimeout(showSlides, 2000); // Change image every 2 seconds
                            }
                            </script>

                            </body>
                            </html>

                        """,
                        height=300,
    )

def select_target_class_column(df, target):
    with st.popover('Select target class'):
        with st.form('target_selection_form'):
            st.write(f'''Current target class: {target}''')
            default_ix = df.columns.tolist().index(target)
            selected_target = st.selectbox('Select target feature', df.columns.tolist(), index=default_ix)

            if st.form_submit_button('Select'):
                st.session_state['data']['target'] = selected_target
                check_feature_dtype(df,target)


def check_nulls(df):
    n_nulls = df.isnull().sum().sum()

    if n_nulls != 0:
        st.session_state['preprocessing']['nulls']['presence'] = True
    else:
        pass

def warning_dataset_load():

    if 'data' not in st.session_state:
        st.write('')
        st.write('')
        st.warning('Please select dataset to load on the sidebar')
        st.stop()

def warning_build_model():
    if 'x_train' not in st.session_state['object']:
        st.write('')
        st.write('')
        st.warning('Please build a model first')
        cols =st.columns([0.5,1,1])
        with cols[1]:
            switch_page_build = st.button('Build a ML model')

        if switch_page_build:
            switch_page('model_training')

        st.stop()

def create_impute_strategy_selector(key):
    '''
        Create select box to chose strategy for filling missing values
    '''
    if key == 'num':
        impute_strategies = ('mean','median')
        return st.selectbox('Impute method', options=impute_strategies, key=key)
    else:
        with st.container():
        # categorical
            st.code('''impute method: 'most_frequent' ''')
            st.info('It replaces missing using the most frequent value along each column')
            return 'most_frequent'

def insert_nan_values(df, target):
    '''
        randomly replace 10% of data with nan to examine the missing value imputation
    '''

    cols = df.drop(target, axis=1).columns.tolist()
    for idx in cols:
        df[idx] = [item if np.random.rand() < 0.9
                                    else None for item in df[idx]]

    # update  the session_state['preprocessing']['nulls']['presence'] -> True
    check_nulls(df)
    return df

def select_feature_dtypes(key):
    '''
        create a list of num or cat features included in model
    '''

    features = st.session_state['data'][f'{key}_features']
    excluded_features = st.session_state['model']['features']['excluded']

    if excluded_features is None:
        return features
    else:
        return [feature for feature in features if feature not in excluded_features]

def feature_scaler(key):

    if key == 'Min-Max scaling':
        return MinMaxScaler()
    elif key == 'Z-score normalization (Standardization)':
        return StandardScaler()
    else:
        return StandardScaler()

def categorical_feature_encoder(key):
    'add  otions if you test other encoders'

    if key == 'OneHotEncoder':
        return OneHotEncoder(handle_unknown='ignore')

    else:
        return OneHotEncoder(handle_unknown='ignore')

# set_config menu for pipeline display
set_config(display='diagram')

from sklearn.pipeline import make_pipeline

def construct_pipeline():
    '''construct pipeline for preprocessing and transformation
        1. obtain num/cat features included in a model
        2. use simple imputer to replace nan values
        3. data normalization for numerical features
        4. One hot encoding for categorical features
    '''

    num_features = select_feature_dtypes('num')
    cat_features = select_feature_dtypes('cat')

    num_impute_strategy = st.session_state['preprocessing']['nulls']['strategy']['num_features']
    cat_impute_strategy = st.session_state['preprocessing']['nulls']['strategy']['cat_features']

    perform_scaling = st.session_state['preprocessing']['normalization']['scaling']
    normalizer = st.session_state['preprocessing']['normalization']['scaler']

    cat_encoder = st.session_state['preprocessing']['encoding']['encoder']

    if perform_scaling:
        num_pipe = Pipeline(steps=[
            ('num_imputer', SimpleImputer(strategy=num_impute_strategy)),
            ('scaler', feature_scaler(normalizer))
        ])
    else:
        num_pipe = Pipeline(steps=[
            ('num_imputer', SimpleImputer(strategy=num_impute_strategy)),
        ])

    cat_pipe = Pipeline([
        ('cat_imputer', SimpleImputer(strategy=cat_impute_strategy)),
        ('encoding', categorical_feature_encoder(cat_encoder))
    ])

    ct = ColumnTransformer(
        transformers=[
            ('numerical_features', num_pipe, num_features),
            ('categorical_features', cat_pipe, cat_features)
        ],remainder='passthrough')

    algorithm = st.session_state['model']['algorithm']
    params = st.session_state['model']['params']

    pipe = Pipeline(steps=[('column_transformer', ct),
                           ('classifier', select_ml_algorithm(algorithm, params))])

    # save pipe diagram as html to inspect
    path_to_html = './data/pipeline_estimator.html'
    with open(path_to_html, 'w') as f:
        f.write(estimator_html_repr(pipe))

    with open(path_to_html,'r') as f:
        html_data = f.read()

    st.markdown('<b>Visualize a scikit-learn pipeline in the model</b>', unsafe_allow_html=True)
    cols = st.columns([0.1,2])
    with cols[1]:
        st.components.v1.html(html_data, scrolling=True, height=300)

    return pipe

def save_classification_report(df):

    file_name = "./data/classification_report.csv"
    if os.path.isfile(file_name):

        expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".csv")[0] + str(expand) + ".csv"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break

    df.to_csv(file_name)

@st.cache_data
def load_model():

    model_path = './data/ml_model.model'
    model = load(model_path)

    return model





