#import libraries
import pandas as pd
import plotly.express as px
import numpy as np
import warnings
import streamlit as st
import time

from utils import make_sidebar, cover_page, select_target_class_column
from plot_func import plot_scatter_matrix, generate_heatmap

warnings.filterwarnings('ignore')

st.set_page_config(
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

make_sidebar()
cover_page()


# st.session_state['test'] = {'fruit': {'include': None,
#                                       'exclude': None}

#     }

# st.write(st.session_state['test']['fruit']['include'])
# st.session_state['test']['fruit']['include'] = [2]
# st.write(st.session_state)

# st.stop()





if 'data' not in st.session_state:
    st.write('')
    st.write('')
    st.warning('Please select dataset to load on the sidebar')
    st.stop()

df = st.session_state['data']['dataframe']
target_class_names = st.session_state['data']['target_class_names']
target = st.session_state['data']['target']

st.header('Data', divider='orange')

cols = st.columns([0.2,0.5,2,0.5])
with cols[0]:
    st.write('')
with cols[1]:
    select_target_class_column(df, target)
    target = st.session_state['data']['target']

with cols[2]:
    st.info('''Default: 'target' or the last column name in dataset''')
with cols[3]:
    st.write('')

tabs = st.tabs(['Column Summary', 'Data Table'])

with tabs[0]:
    with st.container():
        with st.expander('Table summary'):
            cols = st.columns([0.1,2])
            with cols[0]:
                st.write('')
            with cols[1]:
                table_summary = df.describe()
                table_summary.loc['Nulls'] = df.isnull().sum()
                st.write(table_summary)

    for i, col_name in enumerate(df.columns):
        with st.container():
            with st.expander(col_name):
                cols = st.columns([1,4,3,3])
                with cols[0]:
                    st.write('')
                with cols[1]:
                    if st.checkbox('Show target class',key=i):
                        color = target
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

    if st.session_state['data']['dataset'] == 'Iris':
        st.plotly_chart(plot_scatter_matrix(df, target, target_class_names))

    else:
        st.info('Including more than 8 features makes a graph difficult to observe data points')
        with st.form('Scatter_matrix_plot_large_features'):
            selected_features = st.multiselect('Select features for scatterplot matrix',
                                        options=df.drop(columns=target).columns.tolist(),
                                        default=df.drop(columns=target).columns.tolist()[:5],
                                        )

            if st.form_submit_button('Run'):
                selected_features_plot = selected_features
            else:
                selected_features_plot = None

        if selected_features_plot:
            st.plotly_chart(plot_scatter_matrix(df, target, target_class_names, selected_features_plot))
with col2:
    st.subheader('Correlation plot')
    st.write('compute the coreraltion coefficient')
    st.pyplot(generate_heatmap(df, target))

st.header('Data Pre-processing', divider='orange')
with st.form('Pre-processing'):
# ----------- Feature selection -----------------------------------
    st.markdown('<b>1. Feature Selection</b>', unsafe_allow_html=True)
    with st.expander('Feature Selection'):
        include_options = st.multiselect(
        'Features included in the model',
        df.drop(target, axis=1).columns.tolist(),
        df.drop(target, axis=1).columns.tolist()[:10])

        exclude_options = [x for x in df.drop(target, axis=1).columns.tolist() if x not in include_options]

        st.session_state['model']['features']['included'] = include_options
        st.session_state['model']['features']['excluded'] = exclude_options

# -------------- Missing data handling  -------------------------------------
    st.markdown('<b>2. Missing Data Handling</b>', unsafe_allow_html=True)

    if st.session_state['pre_processing']['nulls']['presence']:
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
                    st.session_state['pre_processing']['nulls']['impute'] = True
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
                        st.session_state['pre_processing']['nulls']['features'] = {col : impute_strategy}
                    with form_cols[2]:
                        fill_missing_constant = st.number_input('Constant to fill missing values',
                                                                key=f'missing_value_constant_input_{col}',
                                                                help='If imputation strategy constant is selected')
                        if impute_strategy == 'constant':
                            st.session_state['pre_processing']['normalization']['constant'] = fill_missing_constant
                        else:
                            pass
                    st.divider()

    else:
        st.info('There is no missing value')


#  ----------- Data Normalization -----------------
    st.markdown('<b>3. Data Normalization </b>', unsafe_allow_html=True)

    with st.expander('Data Normalization'):
        if st.checkbox('Perform data normalization'):
            st.session_state['pre_processing']['normalization']['scaling'] = True
        numeric_features = df.drop(target, axis=1).select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.drop(target, axis=1).select_dtypes(include='object').columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Numeric features')
            selected_scaler = st.selectbox('Select normalization method',('Min-Max scaling', 'Z-score normalization (Standardization)'))
            st.session_state['pre_processing']['normalization']['scaler'] = selected_scaler
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
            st.session_state['pre_processing']['normalization']['features'] = select_features_scaled

        with col2:
            st.subheader('Categorical features')

            if categorical_features:

                select_features_hotencoded = st.multiselect('Select features to be hot encoded', categorical_features, categorical_features)
                st.session_state['pre_processing']['hot_encoding']['features'] = select_features_hotencoded

            else:
                st.write('')
                st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;No categorical features are found', unsafe_allow_html=True)

        st.info('Currently the same normalization/hot-encoding methods are applied to all selected numeric/categorical features. It might be benefitial to add a feature to change normalization method per feature')

    submitted = st.form_submit_button('Apply')

    if submitted:
        with st.container():
            cols =st.columns([0.5,4,1])
            with cols[0]:
                st.write('')
            with cols[1]:

                present_nulls = st.session_state['pre_processing']['nulls']['presence']
                impute = st.session_state['pre_processing']['nulls']['impute']

                if (present_nulls) & (impute != True):
                    st.warning('Null values in your dataset may impact on performance of your machine learning model')

                model_includes_features = st.session_state['model']['features']['included']
                model_excludes_features = st.session_state['model']['features']['excluded']
                apply_scaler = st.session_state['pre_processing']['normalization']['scaling']
                scaler = st.session_state['pre_processing']['normalization']['scaler']

                st.code(f"""
                            Shape of predictor dataset  :   {df.drop(columns=target).shape} \n
                            Target column name          :   {target}\n
                            Number of classes           :   {len(np.unique(df[target]))} \n
                            Target class                :   {(*target_class_names,)} \n
                            Features included           :   {len(model_includes_features)} features, {model_includes_features}\n
                            Features excluded           :   {len(model_excludes_features)} features, {model_excludes_features}\n
                            Missing values              :   {present_nulls}\n
                            Imputation                  :   {impute}\n
                            Normalization               :   {apply_scaler}     {scaler if apply_scaler else ''}\n
                        """)
            with cols[2]:
                st.write('')

# # -------------------------------------------------
# # prediction with your input
# # https://builtin.com/data-science/feature-importance



