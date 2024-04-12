#import libraries
import plotly.express as px
import warnings
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from utils import make_sidebar, local_css, warning_dataset_load, select_target_class_column
from plot_func import plot_scatter_matrix, generate_heatmap

warnings.filterwarnings('ignore')

# load local css for sidebar
local_css('./style.css')
make_sidebar()

warning_dataset_load()

df = st.session_state['object']['dataframe']
target_class_names = st.session_state['data']['target_class_names']
target = st.session_state['data']['target']

st.title('Exploratory Data Analysis', anchor='EDA')
st.write('')
st.header('Data Table and Column summary', anchor='data-table-header',divider='orange')

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

st.header('Data Inspection', anchor='data-inspection-header', divider='orange')
presence_nulls = st.session_state['preprocessing']['nulls']['presence']

if presence_nulls:
    st.warning('The dataset contains null values')
else:
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

st.write('')
st.write('')

st.divider()
cols = st.columns([1,0.5,1])
with cols[1]:
    st.subheader('to a model building page')
cols = st.columns([1,0.6,0.9])
with cols[1]:
    switch_page_build = st.button('Build a ML model')

if switch_page_build:
    switch_page('model_training')
