"""
Visualization module for synthetic data analysis.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def create_categorical_pie_chart(df, column_name):

    counts = df[column_name].value_counts().head(10)  # Limit to top 10
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title=f"{column_name} Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return fig


def create_numeric_histogram(df, column_name):

    fig = px.histogram(
        df,
        x=column_name,
        nbins=50,
        title=f"{column_name} Distribution",
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(showlegend=False)
    return fig


def create_numeric_boxplot(df, column_name):

    fig = px.box(
        df,
        y=column_name,
        title=f"{column_name} Distribution",
        color_discrete_sequence=['#EF553B']
    )
    return fig


def create_categorical_bar_chart(df, column_name):

    counts = df[column_name].value_counts()
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title=f"{column_name} Distribution",
        labels={'x': column_name, 'y': 'Count'},
        color=counts.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False)
    return fig


def create_correlation_heatmap(df, numeric_columns):

    corr_matrix = df[numeric_columns].corr()
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlations",
        color_continuous_scale='RdBu_r',
        aspect='auto',
        labels=dict(color="Correlation")
    )
    return fig


def generate_data_visualizations(df):

    visualizations = {}
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Visualization 1: First categorical column pie chart
    if categorical_cols:
        first_cat_col = categorical_cols[0]
        visualizations[f'{first_cat_col}_pie'] = {
            'title': f"{first_cat_col} Distribution",
            'figure': create_categorical_pie_chart(df, first_cat_col)
        }
    
    # Visualization 2: First numeric column histogram
    if len(numeric_cols) > 0:
        first_num_col = numeric_cols[0]
        visualizations[f'{first_num_col}_hist'] = {
            'title': f"{first_num_col} Distribution",
            'figure': create_numeric_histogram(df, first_num_col)
        }
    
    # # Visualization 3: Second numeric column box plot
    # if len(numeric_cols) > 1:
    #     second_num_col = numeric_cols[1]
    #     visualizations[f'{second_num_col}_box'] = {
    #         'title': f"{second_num_col} Distribution",
    #         'figure': create_numeric_boxplot(df, second_num_col)
    #     }
    
    # # Visualization 4: Second categorical column bar chart
    # if len(categorical_cols) > 1:
    #     second_cat_col = categorical_cols[1]
    #     visualizations[f'{second_cat_col}_bar'] = {
    #         'title': f"{second_cat_col} Distribution",
    #         'figure': create_categorical_bar_chart(df, second_cat_col)
    #     }
    
    # # Visualization 5: Correlation heatmap
    # if len(numeric_cols) > 2:
    #     visualizations['correlation_heatmap'] = {
    #         'title': "Correlation Heatmap",
    #         'figure': create_correlation_heatmap(df, numeric_cols)
    #     }
    
    return visualizations
