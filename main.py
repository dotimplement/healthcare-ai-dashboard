import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Repository Dashboard",
    page_icon="ðŸ¥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the healthcare repository data"""
    df = pd.read_csv('healthcare_data.csv')
    
    # Remove unimportant categories
    df = df[~df['Category'].isin(['Lists', 'Tutorials', 'Specification / Docs', 
                                   'Archived / Deprecated', 'Educational'])]
    
    # Convert to datetime
    df['Created'] = pd.to_datetime(df['Created'])
    df['Last Commit'] = pd.to_datetime(df['Last Commit'])
    
    # Create new columns
    today = datetime.now()
    df['days_since_last_commit'] = (today - df['Last Commit']).apply(lambda x: x.days)
    df['contributor_count'] = df['Top Contributors'].fillna('').str.split(',').apply(len)
    df['Org'] = df['Repository'].str.split('/').str[0]
    # TODO: Make recency days a user defined input
    df['recent_activity_category'] = np.where(df['days_since_last_commit'] < 365, 'Active', 'Inactive')
    
    # Additional processing for visualizations
    df['first_commit'] = df['Created']
    df['last_commit'] = df['Last Commit']
    df['lifespan_days'] = (df['last_commit'] - df['first_commit']).dt.days
    df['start_year'] = df['first_commit'].dt.year
    df['is_active'] = df['last_commit'] >= (today - timedelta(days=180))
    
    return df

# Load data
try:
    df = load_and_process_data()
    
    # Title and introduction
    st.markdown('<p class="main-header">Healthcare AI Repository Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Explore insights from healthcare AI repositories including stars, activity, and top contributors.")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=sorted(df['Category'].unique()),
        default=sorted(df['Category'].unique())
    )
    
    activity_filter = st.sidebar.radio(
        "Repository Activity",
        options=["All", "Active Only", "Inactive Only"],
        index=0
    )
    
    # TODO: MAke a double slider to set both min and max stars
    min_stars = st.sidebar.slider(
        "Minimum Stars",
        min_value=0,
        max_value=int(df['Stars'].max()),
        value=0
    )
    
    # Apply filters
    filtered_df = df[df['Category'].isin(selected_categories)]
    if activity_filter == "Active Only":
        filtered_df = filtered_df[filtered_df['recent_activity_category'] == 'Active']
    elif activity_filter == "Inactive Only":
        filtered_df = filtered_df[filtered_df['recent_activity_category'] == 'Inactive']
    filtered_df = filtered_df[filtered_df['Stars'] >= min_stars]
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Repositories", len(filtered_df))
    with col2:
        st.metric("Total Stars", f"{filtered_df['Stars'].sum():,}")
    with col3:
        st.metric("Active Repos", len(filtered_df[filtered_df['recent_activity_category'] == 'Active']))
    with col4:
        st.metric("Unique Organizations", filtered_df['Org'].nunique())
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Dataset Visuals", "Top Contributors", "Temporal Analysis"])
    
    # TAB 1: Dataset Visuals
    with tab1:
        st.markdown('<p class="sub-header">Distribution Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stars by Category
            fig1 = px.box(
                filtered_df, 
                x='Category', 
                y='Stars', 
                title='Distribution of Stars by Category',
                color='Category'
            )
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Category counts
            category_counts = filtered_df['Category'].value_counts()
            fig2 = px.bar(
                x=category_counts.index, 
                y=category_counts.values,
                labels={'x': 'Category', 'y': 'Count'},
                title='Repository Count by Category',
                color=category_counts.values,
                color_continuous_scale='Blues'
            )
            fig2.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Stars by Category (Log Scale)
            fig3 = px.box(
                filtered_df, 
                x='Category', 
                y='Stars', 
                log_y=True,
                title='Distribution of Stars by Category (Log Scale)',
                color='Category'
            )
            fig3.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Language distribution
            language_counts = filtered_df['Language'].value_counts().head(10)
            fig4 = px.bar(
                x=language_counts.index, 
                y=language_counts.values,
                labels={'x': 'Language', 'y': 'Count'},
                title='Top 10 Programming Languages',
                color=language_counts.values,
                color_continuous_scale='Viridis'
            )
            fig4.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Category and Subcategory breakdown
        st.markdown('<p class="sub-header">Category Hierarchy</p>', unsafe_allow_html=True)
        counts_df = filtered_df.groupby(['Category', 'Subcat']).size().reset_index(name='count')
        fig5 = px.sunburst(
            counts_df, 
            path=['Category', 'Subcat'], 
            values='count',
            title='Category and Subcategory Hierarchy',
            height=600
        )
        st.plotly_chart(fig5, use_container_width=True)
        
        # Parallel categories
        st.markdown('<p class="sub-header">Multi-dimensional Analysis</p>', unsafe_allow_html=True)
        parallel_df = filtered_df[(filtered_df['Stars'] < 500) & (filtered_df['Stars'] > 50)]
        if len(parallel_df) > 0:
            fig6 = px.parallel_categories(
                parallel_df,
                dimensions=['Category', 'Subcat', 'recent_activity_category'],
                color='Stars',
                color_continuous_scale='Turbo',
                title='Repository Categories - Parallel View'
            )
            fig6.update_layout(height=600)
            st.plotly_chart(fig6, use_container_width=True)
    
    # TAB 2: Top Contributors
    with tab2:
        st.markdown('<p class="sub-header">Top Contributors Explorer</p>', unsafe_allow_html=True)
        
        # Extract all contributors
        contributors_list = []
        for idx, row in filtered_df.iterrows():
            if pd.notna(row['Top Contributors']):
                contributors = str(row['Top Contributors']).split(', ')
                for contributor in contributors:
                    contributor_clean = contributor.strip()
                    # Filter out Dependabot and similar bot accounts
                    if contributor_clean.lower() not in ['dependabot', 'dependabot[bot]', 'dependabot-preview[bot]']:
                        contributors_list.append({
                            'Contributor': contributor_clean,
                            'Repository': row['Repository'],
                            'Stars': row['Stars'],
                            'Category': row['Category'],
                            'Org': row['Org'],
                            'Active': row['recent_activity_category']
                        })
        
        contributors_df = pd.DataFrame(contributors_list)
        
        if len(contributors_df) > 0:
            # Top contributors by total stars
            top_contributors = contributors_df.groupby('Contributor').agg({
                'Stars': 'sum',
                'Repository': 'count'
            }).reset_index()
            top_contributors.columns = ['Contributor', 'Total Stars', 'Repo Count']
            top_contributors = top_contributors.sort_values('Total Stars', ascending=False).head(20)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Top 20 Contributors by Total Stars")
                fig7 = px.bar(
                    top_contributors,
                    x='Total Stars',
                    y='Contributor',
                    orientation='h',
                    title='Top Contributors by Total Stars',
                    color='Total Stars',
                    color_continuous_scale='Blues',
                    hover_data=['Repo Count']
                )
                fig7.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig7, use_container_width=True)
            
            with col2:
                st.markdown("#### Top 20 Contributors by Repository Count")
                top_by_count = contributors_df.groupby('Contributor').agg({
                    'Repository': 'count',
                    'Stars': 'sum'
                }).reset_index()
                top_by_count.columns = ['Contributor', 'Repo Count', 'Total Stars']
                top_by_count = top_by_count.sort_values('Repo Count', ascending=False).head(20)
                
                fig8 = px.bar(
                    top_by_count,
                    x='Repo Count',
                    y='Contributor',
                    orientation='h',
                    title='Top Contributors by Repository Count',
                    color='Repo Count',
                    color_continuous_scale='Greens',
                    hover_data=['Total Stars']
                )
                fig8.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig8, use_container_width=True)
            
            # Contributor search
            st.markdown('<p class="sub-header">Search Contributor</p>', unsafe_allow_html=True)
            search_contributor = st.selectbox(
                "Select a contributor to view details:",
                options=sorted(contributors_df['Contributor'].unique())
            )
            
            if search_contributor:
                contributor_repos = contributors_df[contributors_df['Contributor'] == search_contributor]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Repositories", len(contributor_repos))
                with col2:
                    st.metric("Total Stars", f"{contributor_repos['Stars'].sum():,}")
                with col3:
                    active_count = len(contributor_repos[contributor_repos['Active'] == 'Active'])
                    st.metric("Active Repositories", active_count)
                
                st.markdown("#### Repositories")
                display_repos = contributor_repos[['Repository', 'Stars', 'Category', 'Org', 'Active']].sort_values('Stars', ascending=False)
                st.dataframe(display_repos, use_container_width=True, height=400)
        
        # Top Organizations
        st.markdown('<p class="sub-header">Top Organizations</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            org_stars = filtered_df.groupby('Org')['Stars'].sum().sort_values(ascending=False).head(15)
            fig9 = px.bar(
                x=org_stars.values,
                y=org_stars.index,
                orientation='h',
                title='Top 15 Organizations by Total Stars',
                labels={'x': 'Total Stars', 'y': 'Organization'},
                color=org_stars.values,
                color_continuous_scale='Oranges'
            )
            fig9.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig9, use_container_width=True)
        
        with col2:
            org_count = filtered_df.groupby('Org')['Repository'].count().sort_values(ascending=False).head(15)
            fig10 = px.bar(
                x=org_count.values,
                y=org_count.index,
                orientation='h',
                title='Top 15 Organizations by Repository Count',
                labels={'x': 'Repository Count', 'y': 'Organization'},
                color=org_count.values,
                color_continuous_scale='Purples'
            )
            fig10.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig10, use_container_width=True)
    
    # TAB 3: Temporal Analysis
    with tab3:
        st.markdown('<p class="sub-header">Repository Growth Over Time</p>', unsafe_allow_html=True)
        
        # Cumulative repositories by category
        df_sorted = filtered_df.sort_values('first_commit')
        categories = filtered_df['Category'].unique()
        
        fig11 = go.Figure()
        
        for category in categories:
            category_df = df_sorted[df_sorted['Category'] == category].copy()
            category_df['cumulative_count'] = range(1, len(category_df) + 1)
            
            fig11.add_trace(go.Scatter(
                x=category_df['first_commit'],
                y=category_df['cumulative_count'],
                mode='lines',
                name=category,
                line=dict(width=3)
            ))
        
        fig11.update_layout(
            title='Cumulative Number of Repositories by Category Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Number of Repositories',
            height=600,
            hovermode='closest'
        )
        st.plotly_chart(fig11, use_container_width=True)
        
        # Survival rate analysis
        st.markdown('<p class="sub-header">Repository Survival Analysis</p>', unsafe_allow_html=True)
        
        survival_by_year = filtered_df.groupby('start_year').agg({
            'is_active': ['sum', 'count']
        }).round(3)
        survival_by_year.columns = ['active_repos', 'total_repos']
        survival_by_year['survival_rate'] = (survival_by_year['active_repos'] / survival_by_year['total_repos'] * 100).round(1)
        survival_by_year = survival_by_year.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig12 = px.line(
                survival_by_year,
                x='start_year',
                y='survival_rate',
                markers=True,
                title='Repository Survival Rate by Start Year (%)',
                labels={'start_year': 'Year Started', 'survival_rate': 'Survival Rate (%)'}
            )
            fig12.update_layout(yaxis=dict(range=[0, 100]), height=400)
            st.plotly_chart(fig12, use_container_width=True)
        
        with col2:
            fig13 = px.bar(
                survival_by_year,
                x='start_year',
                y=['active_repos', 'total_repos'],
                title='Active vs Total Repositories by Start Year',
                labels={'value': 'Number of Repos', 'variable': 'Status'},
                barmode='group'
            )
            fig13.update_layout(height=400)
            st.plotly_chart(fig13, use_container_width=True)
        
        # Repository lifespan
        st.markdown('<p class="sub-header">Repository Lifespan Distribution</p>', unsafe_allow_html=True)
        fig14 = px.histogram(
            filtered_df,
            x='lifespan_days',
            nbins=50,
            title='Distribution of Repository Lifespans',
            labels={'lifespan_days': 'Lifespan (Days)', 'count': 'Number of Repos'},
            color_discrete_sequence=['#636EFA']
        )
        fig14.update_layout(height=400)
        st.plotly_chart(fig14, use_container_width=True)

except FileNotFoundError:
    st.error("âš ï¸ Could not find 'healthcare_data.csv'. Please ensure the file is in the same directory as this script.")
except Exception as e:
    st.error(f"âš ï¸ An error occurred: {str(e)}")
    st.info("Please check that your data file is properly formatted and all required columns are present.")