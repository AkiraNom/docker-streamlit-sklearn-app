import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import seaborn as sns

def replace_target_class_int_str(df, target, target_class_name):
    replacement = {x : target_class_name[idx] for idx, x in enumerate(df[target].unique())}
    df[target] = df[target].replace(replacement)

    return df

@st.cache_data
def plot_scatter_matrix(df, target, target_class_names, selected_features=False):
    df = replace_target_class_int_str(df,target,target_class_names)

    if selected_features:

        df = df.loc[:,(st.session_state['feature_plotted']+[st.session_state['target']])]

    fig = px.scatter_matrix(
        data_frame=df,
        dimensions=df.drop(columns=target).columns.tolist(),
        color=target,
        symbol=target
        )
    fig.update_traces(diagonal_visible=False,  showupperhalf=False)
    fig.update_layout(font=dict(size=16))

    return fig

@st.cache_data
def generate_heatmap(df, target):
    corr = df.drop(target,axis=1).corr()
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(corr,ax=ax)

    return fig
