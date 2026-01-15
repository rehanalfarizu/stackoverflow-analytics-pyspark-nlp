"""
Streamlit Dashboard
===================
Interactive dashboard untuk Stack Overflow Analytics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_sample_data():
    """Load sample data untuk demo."""
    # Sample tag trends data
    dates = pd.date_range(start='2020-01-01', periods=48, freq='M')
    
    tags_data = []
    for tag in ['python', 'javascript', 'java', 'rust', 'go']:
        for date in dates:
            import random
            base = {'python': 5000, 'javascript': 4500, 'java': 3000, 
                   'rust': 500, 'go': 800}
            growth = {'python': 1.02, 'javascript': 1.01, 'java': 0.99,
                     'rust': 1.05, 'go': 1.03}
            
            value = base[tag] * (growth[tag] ** (dates.tolist().index(date)))
            value *= (1 + random.uniform(-0.1, 0.1))
            
            tags_data.append({
                'Period': date,
                'Tag': tag,
                'QuestionCount': int(value),
                'AvgScore': random.uniform(1, 5)
            })
    
    tag_trends_df = pd.DataFrame(tags_data)
    
    # Sample quality distribution
    quality_data = pd.DataFrame({
        'Quality': ['High', 'Medium', 'Low'],
        'Count': [15000, 45000, 40000],
        'Percentage': [15, 45, 40]
    })
    
    # Sample sentiment data
    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Count': [35000, 50000, 15000]
    })
    
    # Sample topic data
    topic_data = pd.DataFrame({
        'Topic': [f'Topic {i}' for i in range(1, 11)],
        'DocumentCount': [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500],
        'Keywords': [
            'python, pandas, dataframe',
            'javascript, react, node',
            'java, spring, maven',
            'sql, database, query',
            'css, html, frontend',
            'api, rest, http',
            'docker, kubernetes, container',
            'machine learning, tensorflow',
            'git, github, version control',
            'testing, unit test, pytest'
        ]
    })
    
    return tag_trends_df, quality_data, sentiment_data, topic_data


def create_trend_chart(df, selected_tags):
    """Create trend chart untuk tags."""
    filtered = df[df['Tag'].isin(selected_tags)]
    
    fig = px.line(
        filtered,
        x='Period',
        y='QuestionCount',
        color='Tag',
        title='Technology Trend Over Time',
        labels={'QuestionCount': 'Number of Questions', 'Period': 'Date'}
    )
    
    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def create_quality_pie(df):
    """Create pie chart untuk quality distribution."""
    fig = px.pie(
        df,
        values='Count',
        names='Quality',
        title='Question Quality Distribution',
        color='Quality',
        color_discrete_map={
            'High': '#2ecc71',
            'Medium': '#f39c12',
            'Low': '#e74c3c'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def create_sentiment_bar(df):
    """Create bar chart untuk sentiment."""
    fig = px.bar(
        df,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        title='Comment Sentiment Distribution',
        color_discrete_map={
            'Positive': '#27ae60',
            'Neutral': '#3498db',
            'Negative': '#c0392b'
        }
    )
    
    return fig


def create_topic_bar(df):
    """Create horizontal bar chart untuk topics."""
    fig = px.bar(
        df,
        y='Topic',
        x='DocumentCount',
        orientation='h',
        title='Top Topics by Document Count',
        text='Keywords',
        color='DocumentCount',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig


def create_heatmap(df):
    """Create heatmap untuk activity patterns."""
    # Create sample hourly data
    hours = list(range(24))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    import numpy as np
    np.random.seed(42)
    
    # Create realistic pattern (more activity during work hours, weekdays)
    z = []
    for day_idx, day in enumerate(days):
        row = []
        for hour in hours:
            # Base activity
            base = 100
            
            # Work hours boost (9-17)
            if 9 <= hour <= 17:
                base *= 2
            
            # Weekday boost
            if day_idx < 5:
                base *= 1.5
            
            # Add noise
            value = base * (1 + np.random.uniform(-0.3, 0.3))
            row.append(int(value))
        z.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=hours,
        y=days,
        colorscale='YlOrRd'
    ))
    
    fig.update_layout(
        title='Question Posting Activity Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week'
    )
    
    return fig


def run_dashboard():
    """Run Streamlit dashboard."""
    
    # Page config
    st.set_page_config(
        page_title="Stack Overflow Analytics",
        page_icon="",
        layout="wide"
    )
    
    # Title
    st.title("Stack Overflow Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    tag_trends_df, quality_data, sentiment_data, topic_data = load_sample_data()
    
    # Sidebar
    st.sidebar.header("Filters")
    
    # Date range
    st.sidebar.subheader("Date Range")
    min_date = tag_trends_df['Period'].min()
    max_date = tag_trends_df['Period'].max()
    
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Tag selection
    st.sidebar.subheader("Technologies")
    all_tags = tag_trends_df['Tag'].unique().tolist()
    selected_tags = st.sidebar.multiselect(
        "Select technologies",
        options=all_tags,
        default=all_tags[:3]
    )
    
    # Main content
    # Row 1: KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Questions",
            value="23.5M",
            delta="+125K this month"
        )
    
    with col2:
        st.metric(
            label="Active Users",
            value="1.2M",
            delta="+15K"
        )
    
    with col3:
        st.metric(
            label="Avg Response Time",
            value="2.3 hours",
            delta="-15 min"
        )
    
    with col4:
        st.metric(
            label="Answer Rate",
            value="78.5%",
            delta="+2.1%"
        )
    
    st.markdown("---")
    
    # Row 2: Trend and Quality
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Technology Trends")
        if selected_tags:
            trend_chart = create_trend_chart(tag_trends_df, selected_tags)
            st.plotly_chart(trend_chart, use_container_width=True)
        else:
            st.warning("Please select at least one technology")
    
    with col2:
        st.subheader("Question Quality")
        quality_chart = create_quality_pie(quality_data)
        st.plotly_chart(quality_chart, use_container_width=True)
    
    # Row 3: Sentiment and Activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Comment Sentiment")
        sentiment_chart = create_sentiment_bar(sentiment_data)
        st.plotly_chart(sentiment_chart, use_container_width=True)
    
    with col2:
        st.subheader("Activity Patterns")
        heatmap = create_heatmap(tag_trends_df)
        st.plotly_chart(heatmap, use_container_width=True)
    
    # Row 4: Topics
    st.subheader("Discovered Topics")
    topic_chart = create_topic_bar(topic_data)
    st.plotly_chart(topic_chart, use_container_width=True)
    
    # Row 5: Data Table
    st.subheader("Top Trending Technologies")
    
    # Calculate growth
    latest = tag_trends_df.groupby('Tag').last().reset_index()
    earliest = tag_trends_df.groupby('Tag').first().reset_index()
    
    growth_df = latest[['Tag', 'QuestionCount']].merge(
        earliest[['Tag', 'QuestionCount']],
        on='Tag',
        suffixes=('_latest', '_earliest')
    )
    growth_df['Growth'] = (
        (growth_df['QuestionCount_latest'] - growth_df['QuestionCount_earliest']) /
        growth_df['QuestionCount_earliest'] * 100
    )
    growth_df = growth_df.sort_values('Growth', ascending=False)
    
    st.dataframe(
        growth_df[['Tag', 'QuestionCount_latest', 'Growth']].rename(columns={
            'QuestionCount_latest': 'Current Questions',
            'Growth': 'Growth (%)'
        }),
        use_container_width=True
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Dashboard for Stack Overflow Analytics | "
        "Data Source: Stack Overflow Data Dump"
    )


if __name__ == "__main__":
    run_dashboard()
