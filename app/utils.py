import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,ShuffleSplit,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

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
    target_class_name = data.target_names

    df = pd.DataFrame(feature_data, columns = columns)
    df.loc[:,'target'] = target.reshape(-1,1)

    return df, target_class_name

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

def make_sidebar():
    with st.sidebar:
        st.title('💎 Menu')
        st.write('')
        st.write('')

        tab1, tab2, tab3 = st.tabs(['Data :clipboard:', 'Model parameters', 'Help'])

        with tab1:

            selected_dataset = st.selectbox('Select Dataset',('Iris','Breast Cancer','Wine Dataset'))

            st.subheader(f'{selected_dataset} dataset selected')

            st.session_state['dataset'] = selected_dataset

            selected_algorithm = st.selectbox('Select Algorithm',('Logistic Regression','Random Forests','GBC','Decision Tree','KNN','Support Vector Machines'))

            st.subheader(f'{selected_algorithm} algorithm selected')

            st.session_state['algorithm'] = selected_algorithm

            if st.checkbox('random_state on'):
                selected_random_state = st.number_input('Type random state', step=1)
                st.session_state['random_state'] = selected_random_state
            else:
                st.session_state['random_state'] = None

        with tab2:

            st.write('')
            st.subheader('Data Split')
            with st.container(border=True):
                st.session_state['test_size'] = st.slider(label='Test Data Size', min_value=0.01, max_value=0.99, value=0.3, step=0.1)
                st.markdown(f'''<b>Training data </b>: {1- st.session_state['test_size']}</style>''', unsafe_allow_html=True)
                st.markdown(f'''<b>Testing data </b>: {st.session_state['test_size']}''', unsafe_allow_html=True)
            st.divider()
            st.subheader('Tune model parameters')
            st.session_state['params'] = optimize_hyperparameters(st.session_state['algorithm'])

        with tab3:

            st.write('short description of each dataset')

@st.cache_data
def load_dataset_description(selected_dataset):
    if selected_dataset=="Iris":
        data = datasets.load_iris()
    elif selected_dataset=="Breast Cancer":
        data = datasets.load_breast_cancer()
    elif selected_dataset=="Wine Dataset":
        data = datasets.load_wine()


def cover_page():

    st.header('Iris Classification')

    col1,col2,col3 = st.columns([1,4,1])

    with col1:
        st.write('')
    with col2:
        st.image('https://miro.medium.com/v2/resize:fit:1400/format:webp/0*Uw37vrrKzeEWahdB')
    with col3:
        st.write('')
