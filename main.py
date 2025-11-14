import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Repository Dashboard",
    page_icon="🏥",
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
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Bot account list (manually maintained)
BOT_ACCOUNTS = [
    'dependabot',
    'dependabot[bot]',
    'dependabot-preview[bot]',
    'github-actions[bot]',
    'renovate[bot]',
    'greenkeeper[bot]',
    'imgbot[bot]',
    'allcontributors[bot]',
    'semantic-release-bot',
    'snyk-bot',
    'codecov[bot]',
    'netlify[bot]',
    'whitesource-bolt-for-github[bot]'
]

# Standard organizations to exclude from org charts
STANDARD_ORGS_TO_EXCLUDE = [
    'smart-on-fhir',
    'OHDSI',
    'fhir',
    'HL7',
    'LHNCBC',
    'hl7',
    'ohdsi'
]

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
    
    # Active definition: committed within last 365 days
    df['recent_activity_category'] = np.where(df['days_since_last_commit'] < 365, 'Active', 'Inactive')
    
    # Additional processing for visualizations
    df['first_commit'] = df['Created']
    df['last_commit'] = df['Last Commit']
    df['lifespan_days'] = (df['last_commit'] - df['first_commit']).dt.days
    df['start_year'] = df['first_commit'].dt.year
    df['is_active'] = df['last_commit'] >= (today - timedelta(days=365))
    
    # Process Standard column - handle multiple standards
    df['Standard'] = df['Standard'].fillna('None/Unknown')
    df['has_standard'] = df['Standard'] != 'None/Unknown'
    
    # Create exploded dataframe for multi-standard analysis
    df['standards_list'] = df['Standard'].apply(
        lambda x: [s.strip() for s in str(x).split(',') if s.strip()]
    )
    
    # Determine if org or individual (>3 repos = org)
    org_repo_counts = df.groupby('Org').size()
    df['is_organization'] = df['Org'].map(lambda x: org_repo_counts.get(x, 0) > 3)
    df['owner_type'] = df['is_organization'].map({True: 'Organization', False: 'Individual'})
    
    return df

@st.cache_data
def load_contribution_data():
    """Load contributor statistics from GitHub analysis"""
    try:
        contrib_df = pd.read_csv('contributor_detailed_stats_aggregated.csv')
        return contrib_df
    except FileNotFoundError:
        st.warning("⚠️ Contributor statistics file not found. Top contributors will be ranked by stars instead of lines of code.")
        return None

# Load data
try:
    df = load_and_process_data()
    contrib_stats = load_contribution_data()
    
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
    
    min_stars = st.sidebar.slider(
        "Minimum Stars",
        min_value=0,
        max_value=int(df['Stars'].max()),
        value=0
    )
    
    # Standard filter
    if 'Standard' in df.columns:
        standard_options = ['All'] + sorted(df['Standard'].unique().tolist())
        selected_standard = st.sidebar.selectbox(
            "Filter by Standard",
            options=standard_options,
            index=0
        )
    
    # Apply filters
    filtered_df = df[df['Category'].isin(selected_categories)]
    if activity_filter == "Active Only":
        filtered_df = filtered_df[filtered_df['recent_activity_category'] == 'Active']
    elif activity_filter == "Inactive Only":
        filtered_df = filtered_df[filtered_df['recent_activity_category'] == 'Inactive']
    filtered_df = filtered_df[filtered_df['Stars'] >= min_stars]
    
    if 'Standard' in df.columns and selected_standard != 'All':
        filtered_df = filtered_df[filtered_df['Standard'] == selected_standard]
    
    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Repositories", len(filtered_df))
    with col2:
        st.metric("Total Stars", f"{filtered_df['Stars'].sum():,}")
    with col3:
        st.metric("Active Repos", len(filtered_df[filtered_df['recent_activity_category'] == 'Active']))
    with col4:
        st.metric("Unique Organizations", filtered_df['Org'].nunique())
    with col5:
        if 'Standard' in df.columns:
            std_count = filtered_df[filtered_df['has_standard']].shape[0]
            st.metric("With Standards", std_count)
    
    # Add definitions box
    with st.expander("ℹ️ Definitions & Methodology"):
        st.markdown("""
        **Active Repository**: A repository is considered "active" if it has had at least one commit within the last 365 days.
        
        **Survival Rate**: The percentage of repositories created in a given year that are still active today (have commits within the last 365 days).
        
        **Organization vs Individual**: Accounts with more than 3 repositories in this dataset are classified as organizations; others are classified as individuals.
        
        **Top Contributors**: When contribution statistics are available, contributors are ranked by total lines of code added across all repositories.
        """)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Top Repositories", "Category Analysis", "Standards Analysis", "Top Contributors", "Temporal Analysis"])
    
    # TAB 1: Top Repositories
    with tab1:
        st.markdown('<p class="sub-header">Top Starred Repositories by Category</p>', unsafe_allow_html=True)
        
        # Get categories to display
        categories_to_show = sorted(filtered_df['Category'].unique())
        
        # Create a grid layout - 2 columns
        for i in range(0, len(categories_to_show), 2):
            cols = st.columns(2)
            
            for col_idx, col in enumerate(cols):
                if i + col_idx < len(categories_to_show):
                    category = categories_to_show[i + col_idx]
                    
                    with col:
                        category_df = filtered_df[filtered_df['Category'] == category].nlargest(10, 'Stars')
                        
                        if len(category_df) > 0:
                            fig = px.bar(
                                category_df,
                                y='Repository',
                                x='Stars',
                                orientation='h',
                                title=f'Top 10 in {category}',
                                color='Stars',
                                color_continuous_scale='Blues',
                                hover_data=['Subcat', 'Language']
                            )
                            fig.update_layout(
                                height=400,
                                yaxis={'categoryorder': 'total ascending'},
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        # Special highlight: Interoperability
        st.markdown('<p class="sub-header">🌟 Spotlight: Top 10 Interoperability Projects</p>', unsafe_allow_html=True)
        
        interop_df = filtered_df[filtered_df['Category'] == 'Interoperability'].nlargest(10, 'Stars')
        
        if len(interop_df) > 0:
            fig_interop = px.bar(
                interop_df,
                y='Repository',
                x='Stars',
                orientation='h',
                title='Most Starred Interoperability Projects',
                color='Stars',
                color_continuous_scale='Viridis',
                hover_data=['Subcat', 'Language', 'Standard']
            )
            fig_interop.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig_interop, use_container_width=True)
            
            # Show detailed table
            st.markdown("#### Detailed Information")
            display_cols = ['Repository', 'Stars', 'Subcat', 'Language', 'Standard', 'recent_activity_category']
            st.dataframe(
                interop_df[display_cols].sort_values('Stars', ascending=False),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No Interoperability repositories in the current selection.")
    
    # TAB 2: Category Analysis
    with tab2:
        st.markdown('<p class="sub-header">Category Overview</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category counts with numbers
            category_counts = filtered_df['Category'].value_counts()
            fig_cat_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='Repository Distribution by Category',
                hole=0.4
            )
            # Update to show both count and percentage
            fig_cat_pie.update_traces(
                textposition='inside',
                texttemplate='%{label}<br>%{value} (%{percent})'
            )
            fig_cat_pie.update_layout(height=450)
            st.plotly_chart(fig_cat_pie, use_container_width=True)
        
        with col2:
            # Language distribution with numbers
            language_counts = filtered_df['Language'].value_counts().head(10)
            fig_lang_pie = px.pie(
                values=language_counts.values,
                names=language_counts.index,
                title='Top 10 Programming Languages',
                hole=0.4
            )
            fig_lang_pie.update_traces(
                textposition='inside',
                texttemplate='%{label}<br>%{value} (%{percent})'
            )
            fig_lang_pie.update_layout(height=450)
            st.plotly_chart(fig_lang_pie, use_container_width=True)
        
        # Stacked bar chart: Categories over time
        st.markdown('<p class="sub-header">Category Growth Over Time</p>', unsafe_allow_html=True)
        
        category_timeline = filtered_df.groupby([pd.Grouper(key='first_commit', freq='Y'), 'Category']).size().reset_index(name='count')
        category_timeline['year'] = category_timeline['first_commit'].dt.year
        
        fig_cat_timeline = px.bar(
            category_timeline,
            x='year',
            y='count',
            color='Category',
            title='Repository Creation by Category Over Time',
            labels={'count': 'Number of Repositories', 'year': 'Year'},
            barmode='stack'
        )
        fig_cat_timeline.update_layout(height=500)
        st.plotly_chart(fig_cat_timeline, use_container_width=True)
        
        # Subcategory Explorer
        st.markdown('<p class="sub-header">Subcategory Explorer</p>', unsafe_allow_html=True)
        
        selected_category_explorer = st.selectbox(
            "Select a category to explore subcategories:",
            options=['All Categories'] + sorted(filtered_df['Category'].unique().tolist())
        )
        
        if selected_category_explorer == 'All Categories':
            explorer_df = filtered_df
        else:
            explorer_df = filtered_df[filtered_df['Category'] == selected_category_explorer]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Subcategory distribution
            subcat_counts = explorer_df['Subcat'].value_counts()
            fig_subcat = px.bar(
                x=subcat_counts.values,
                y=subcat_counts.index,
                orientation='h',
                title=f'Subcategory Distribution - {selected_category_explorer}',
                labels={'x': 'Number of Repositories', 'y': 'Subcategory'},
                color=subcat_counts.values,
                color_continuous_scale='Teal'
            )
            fig_subcat.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_subcat, use_container_width=True)
        
        with col2:
            # Subcategory by activity
            subcat_activity = explorer_df.groupby(['Subcat', 'recent_activity_category']).size().reset_index(name='count')
            fig_subcat_activity = px.bar(
                subcat_activity,
                x='Subcat',
                y='count',
                color='recent_activity_category',
                title=f'Subcategory Activity Status - {selected_category_explorer}',
                labels={'count': 'Number of Repositories'},
                barmode='stack',
                color_discrete_map={'Active': '#2ecc71', 'Inactive': '#e74c3c'}
            )
            fig_subcat_activity.update_layout(height=500)
            st.plotly_chart(fig_subcat_activity, use_container_width=True)
        
        # Organization vs Individual breakdown by category
        st.markdown('<p class="sub-header">Organization vs Individual Ownership by Category</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            owner_type_by_cat = filtered_df.groupby(['Category', 'owner_type']).size().reset_index(name='count')
            fig_owner_stack = px.bar(
                owner_type_by_cat,
                x='Category',
                y='count',
                color='owner_type',
                title='Repository Ownership Type by Category',
                labels={'count': 'Number of Repositories'},
                barmode='stack',
                color_discrete_map={'Organization': '#3498db', 'Individual': '#e67e22'}
            )
            fig_owner_stack.update_layout(height=450)
            st.plotly_chart(fig_owner_stack, use_container_width=True)
        
        with col2:
            # Percentage breakdown
            owner_type_pct = filtered_df.groupby('Category').apply(
                lambda x: (x['owner_type'] == 'Organization').sum() / len(x) * 100
            ).reset_index(name='org_percentage')
            
            fig_owner_pct = px.bar(
                owner_type_pct,
                x='Category',
                y='org_percentage',
                title='Percentage of Repositories Owned by Organizations',
                labels={'org_percentage': 'Organization Ownership (%)'},
                color='org_percentage',
                color_continuous_scale='Blues'
            )
            fig_owner_pct.update_layout(height=450, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_owner_pct, use_container_width=True)
        
        # Deep dive: Data Models & Validation
        st.markdown('<p class="sub-header">🔍 Deep Dive: Data Models & Validation</p>', unsafe_allow_html=True)
        
        dm_validation_df = filtered_df[
            (filtered_df['Category'] == 'Data Models & Schemas') | 
            (filtered_df['Subcat'].str.contains('Validation', case=False, na=False))
        ]
        
        if len(dm_validation_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Standards distribution in Data Models & Validation
                dm_standards = dm_validation_df.explode('standards_list')
                dm_standards = dm_standards[dm_standards['standards_list'].notna()]
                dm_standards = dm_standards[dm_standards['standards_list'] != 'None/Unknown']
                
                if len(dm_standards) > 0:
                    dm_std_counts = dm_standards['standards_list'].value_counts()
                    fig_dm_std = px.pie(
                        values=dm_std_counts.values,
                        names=dm_std_counts.index,
                        title='Standards in Data Models & Validation',
                        hole=0.4
                    )
                    fig_dm_std.update_traces(
                        textposition='inside',
                        texttemplate='%{label}<br>%{value} (%{percent})'
                    )
                    fig_dm_std.update_layout(height=400)
                    st.plotly_chart(fig_dm_std, use_container_width=True)
                else:
                    st.info("No standard information available for Data Models & Validation.")
            
            with col2:
                # Language distribution in Data Models & Validation
                dm_lang_counts = dm_validation_df['Language'].value_counts().head(10)
                fig_dm_lang = px.pie(
                    values=dm_lang_counts.values,
                    names=dm_lang_counts.index,
                    title='Languages in Data Models & Validation',
                    hole=0.4
                )
                fig_dm_lang.update_traces(
                    textposition='inside',
                    texttemplate='%{label}<br>%{value} (%{percent})'
                )
                fig_dm_lang.update_layout(height=400)
                st.plotly_chart(fig_dm_lang, use_container_width=True)
            
            # Top repositories in this subcategory
            st.markdown("#### Top 10 Data Models & Validation Repositories")
            dm_top = dm_validation_df.nlargest(10, 'Stars')[['Repository', 'Stars', 'Language', 'Standard', 'recent_activity_category']]
            st.dataframe(dm_top, use_container_width=True)
        else:
            st.info("No repositories found in Data Models & Validation category.")
    
    # TAB 3: Standards Analysis
    with tab3:
        st.markdown('<p class="sub-header">Data Standards Analysis</p>', unsafe_allow_html=True)
        
        # Explode standards for individual counting
        standards_exploded = filtered_df.explode('standards_list')
        standards_exploded = standards_exploded[standards_exploded['standards_list'].notna()]
        standards_exploded = standards_exploded[standards_exploded['standards_list'] != 'None/Unknown']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual standards count (from exploded data)
            standard_counts = standards_exploded['standards_list'].value_counts()
            fig_std1 = px.pie(
                values=standard_counts.values,
                names=standard_counts.index,
                title='Distribution of Individual Standards (repos may use multiple)',
                hole=0.4
            )
            fig_std1.update_traces(
                textposition='inside',
                texttemplate='%{label}<br>%{value} (%{percent})'
            )
            fig_std1.update_layout(height=450)
            st.plotly_chart(fig_std1, use_container_width=True)
        
        with col2:
            # Standards by repository count
            fig_std2 = px.bar(
                x=standard_counts.values,
                y=standard_counts.index,
                orientation='h',
                title='Repository Count by Individual Standard',
                labels={'x': 'Number of Repositories', 'y': 'Standard'},
                color=standard_counts.values,
                color_continuous_scale='Teal'
            )
            fig_std2.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_std2, use_container_width=True)
        
        # Standards by Category
        st.markdown('<p class="sub-header">Standards Adoption by Category</p>', unsafe_allow_html=True)

        # Stacked bar chart
        standard_category = standards_exploded.groupby(['Category', 'standards_list']).size().reset_index(name='count')
        fig_std4 = px.bar(
            standard_category,
            x='Category',
            y='count',
            color='standards_list',
            title='Repository Count by Category and Standard',
            labels={'count': 'Number of Repositories', 'standards_list': 'Standard'},
            barmode='stack'
        )
        fig_std4.update_layout(height=450)
        st.plotly_chart(fig_std4, use_container_width=True)
        
        # Standards and Stars correlation
        st.markdown('<p class="sub-header">Standards and Repository Popularity</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average stars by standard
            standard_stars = standards_exploded.groupby('standards_list').agg({
                'Stars': ['mean', 'sum', 'count']
            }).round(0)
            standard_stars.columns = ['Avg Stars', 'Total Stars', 'Repo Count']
            standard_stars = standard_stars.reset_index().sort_values('Avg Stars', ascending=False)
            standard_stars.columns = ['Standard', 'Avg Stars', 'Total Stars', 'Repo Count']
            
            fig_std5 = px.bar(
                standard_stars,
                x='Avg Stars',
                y='Standard',
                orientation='h',
                title='Average Stars by Standard',
                color='Avg Stars',
                color_continuous_scale='Blues',
                hover_data=['Total Stars', 'Repo Count']
            )
            fig_std5.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_std5, use_container_width=True)
        
        with col2:
            # Box plot of stars distribution by standard
            fig_std6 = px.box(
                standards_exploded,
                x='standards_list',
                y='Stars',
                title='Stars Distribution by Standard',
                color='standards_list',
                log_y=True,
                labels={'standards_list': 'Standard'}
            )
            fig_std6.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_std6, use_container_width=True)
        
        # Standards adoption over time
        st.markdown('<p class="sub-header">Standards Adoption Timeline</p>', unsafe_allow_html=True)
        
        standard_timeline = standards_exploded.groupby([pd.Grouper(key='first_commit', freq='Y'), 'standards_list']).size().reset_index(name='count')
        standard_timeline['first_commit'] = standard_timeline['first_commit'].dt.year
        standard_timeline.columns = ['Year', 'Standard', 'count']
        
        fig_std7 = px.line(
            standard_timeline,
            x='Year',
            y='count',
            color='Standard',
            title='Repository Creation by Standard Over Time',
            labels={'count': 'Number of Repositories Created'},
            markers=True
        )
        fig_std7.update_layout(height=450)
        st.plotly_chart(fig_std7, use_container_width=True)
    
    # TAB 4: Top Contributors
    with tab4:
        st.markdown('<p class="sub-header">Top Contributors Explorer</p>', unsafe_allow_html=True)
        
        # Extract all contributors (filtering out bots)
        contributors_list = []
        for idx, row in filtered_df.iterrows():
            if pd.notna(row['Top Contributors']):
                contributors = str(row['Top Contributors']).split(', ')
                for contributor in contributors:
                    contributor_clean = contributor.strip()
                    # Filter out bots
                    if contributor_clean.lower() not in [bot.lower() for bot in BOT_ACCOUNTS]:
                        contributors_list.append({
                            'Contributor': contributor_clean,
                            'Repository': row['Repository'],
                            'Stars': row['Stars'],
                            'Category': row['Category'],
                            'Org': row['Org'],
                            'Active': row['recent_activity_category'],
                            'Standard': row.get('Standard', 'None/Unknown')
                        })
        
        contributors_df = pd.DataFrame(contributors_list)
        
        if len(contributors_df) > 0:
            # Merge with contribution statistics if available
            if contrib_stats is not None:
                st.info("📊 Using detailed contribution statistics (lines of code) for ranking.")
                
                # Merge contributors with their stats
                contributors_with_stats = contributors_df.merge(
                    contrib_stats[['username', 'total_additions', 'total_commits', 'total_net_lines']],
                    left_on='Contributor',
                    right_on='username',
                    how='left'
                )
                
                # Fill NaN with 0 for contributors without stats
                contributors_with_stats['total_additions'] = contributors_with_stats['total_additions'].fillna(0)
                contributors_with_stats['total_commits'] = contributors_with_stats['total_commits'].fillna(0)
                contributors_with_stats['total_net_lines'] = contributors_with_stats['total_net_lines'].fillna(0)
                
                # Aggregate by contributor
                top_contributors = contributors_with_stats.groupby('Contributor').agg({
                    'total_additions': 'first',
                    'total_commits': 'first',
                    'Stars': 'sum',
                    'Repository': 'count'
                }).reset_index()
                top_contributors.columns = ['Contributor', 'Lines Added', 'Total Commits', 'Total Stars', 'Repo Count']
                top_contributors = top_contributors.sort_values('Lines Added', ascending=False).head(20)
                
                metric_for_chart = 'Lines Added'
                color_scale = 'Viridis'
            else:
                st.warning("⚠️ Contribution statistics not available. Using stars for ranking.")
                
                # Fallback to stars-based ranking
                top_contributors = contributors_df.groupby('Contributor').agg({
                    'Stars': 'sum',
                    'Repository': 'count'
                }).reset_index()
                top_contributors.columns = ['Contributor', 'Total Stars', 'Repo Count']
                top_contributors = top_contributors.sort_values('Total Stars', ascending=False).head(20)
                
                metric_for_chart = 'Total Stars'
                color_scale = 'Blues'
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### Top 20 Contributors by {metric_for_chart}")
                fig7 = px.bar(
                    top_contributors,
                    x=metric_for_chart,
                    y='Contributor',
                    orientation='h',
                    title=f'Top Contributors by {metric_for_chart}',
                    color=metric_for_chart,
                    color_continuous_scale=color_scale,
                    hover_data=top_contributors.columns.tolist()[1:]
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
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Repositories", len(contributor_repos))
                with col2:
                    st.metric("Total Stars", f"{contributor_repos['Stars'].sum():,}")
                with col3:
                    active_count = len(contributor_repos[contributor_repos['Active'] == 'Active'])
                    st.metric("Active Repositories", active_count)
                with col4:
                    if contrib_stats is not None:
                        user_contrib = contrib_stats[contrib_stats['username'] == search_contributor]
                        if len(user_contrib) > 0:
                            st.metric("Lines Added", f"{int(user_contrib['total_additions'].values[0]):,}")
                        else:
                            st.metric("Lines Added", "N/A")
                
                st.markdown("#### Repositories")
                display_repos = contributor_repos[['Repository', 'Stars', 'Category', 'Standard', 'Org', 'Active']].sort_values('Stars', ascending=False)
                st.dataframe(display_repos, use_container_width=True, height=400)
        
        # Top Organizations (excluding standard organizations)
        st.markdown('<p class="sub-header">Top Organizations</p>', unsafe_allow_html=True)
        
        # Filter out standard organizations
        org_filtered_df = filtered_df[~filtered_df['Org'].isin(STANDARD_ORGS_TO_EXCLUDE)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            org_stars = org_filtered_df.groupby('Org')['Stars'].sum().sort_values(ascending=False).head(15)
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
            org_count = org_filtered_df.groupby('Org')['Repository'].count().sort_values(ascending=False).head(15)
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
    
    # TAB 5: Temporal Analysis
    with tab5:
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
        
        with st.expander("Survival Rate Definition"):
            st.markdown("""
            The survival rate represents the percentage of repositories created in a given year 
            that are still "active" today. A repository is considered active if it has had at least one commit within the last 365 days.
            """)
        
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

        
        # Repository Timeline View
        st.markdown('<p class="sub-header">Repository Timeline View</p>', unsafe_allow_html=True)
        
        # Selection controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            timeline_category = st.selectbox(
                "Select Category for Timeline",
                options=['All'] + sorted(filtered_df['Category'].unique().tolist()),
                key='timeline_category'
            )
        
        with col2:
            sort_option = st.selectbox(
                "Sort repositories by",
                options=['Stars (High to Low)', 'Stars (Low to High)', 
                        'Lifespan (Longest)', 'Lifespan (Shortest)',
                        'Most Recent', 'Oldest'],
                key='timeline_sort'
            )
        
        with col3:
            n_repos = st.slider(
                "Number of repos",
                min_value=5,
                max_value=50,
                value=20,
                key='timeline_n'
            )
        
        # Filter and sort data for timeline
        if timeline_category == 'All':
            timeline_df = filtered_df.copy()
        else:
            timeline_df = filtered_df[filtered_df['Category'] == timeline_category].copy()
        
        # Apply sorting
        if sort_option == 'Stars (High to Low)':
            timeline_df = timeline_df.sort_values('Stars', ascending=False)
        elif sort_option == 'Stars (Low to High)':
            timeline_df = timeline_df.sort_values('Stars', ascending=True)
        elif sort_option == 'Lifespan (Longest)':
            timeline_df = timeline_df.sort_values('lifespan_days', ascending=False)
        elif sort_option == 'Lifespan (Shortest)':
            timeline_df = timeline_df.sort_values('lifespan_days', ascending=True)
        elif sort_option == 'Most Recent':
            timeline_df = timeline_df.sort_values('first_commit', ascending=False)
        else:  # Oldest
            timeline_df = timeline_df.sort_values('first_commit', ascending=True)
        
        timeline_df = timeline_df.head(n_repos)
        
        if len(timeline_df) > 0:
            # Create timeline visualization
            fig_timeline = go.Figure()
            
            # Add a line for each repository
            for idx, row in timeline_df.iterrows():
                # Create hover text with detailed info
                hover_text = (
                    f"<b>{row['Repository']}</b><br>"
                    f"Stars: {row['Stars']:,}<br>"
                    f"Category: {row['Category']}<br>"
                    f"Created: {row['first_commit'].strftime('%Y-%m-%d')}<br>"
                    f"Last Commit: {row['last_commit'].strftime('%Y-%m-%d')}<br>"
                    f"Lifespan: {row['lifespan_days']} days<br>"
                    f"Status: {row['recent_activity_category']}"
                )
                
                # Color based on activity
                line_color = '#2ecc71' if row['recent_activity_category'] == 'Active' else '#e74c3c'
                
                # Add line from creation to last commit
                fig_timeline.add_trace(go.Scatter(
                    x=[row['first_commit'], row['last_commit']],
                    y=[row['Repository'], row['Repository']],
                    mode='lines+markers',
                    line=dict(color=line_color, width=3),
                    marker=dict(size=8, symbol=['circle', 'square']),
                    hovertext=[hover_text, hover_text],
                    hoverinfo='text',
                    showlegend=False
                ))
            
            # Add legend manually
            fig_timeline.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='#2ecc71'),
                name='Active',
                showlegend=True
            ))
            fig_timeline.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color='#e74c3c'),
                name='Inactive',
                showlegend=True
            ))
            
            fig_timeline.update_layout(
                title=f'Repository Lifespans Timeline - {timeline_category}',
                xaxis_title='Date',
                yaxis_title='Repository',
                height=max(400, n_repos * 25),
                hovermode='closest',
                yaxis=dict(
                    tickmode='linear',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Summary statistics for selected repos
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Lifespan", f"{timeline_df['lifespan_days'].mean():.0f} days")
            with col2:
                st.metric("Avg Stars", f"{timeline_df['Stars'].mean():.0f}")
            with col3:
                active_pct = (timeline_df['recent_activity_category'] == 'Active').sum() / len(timeline_df) * 100
                st.metric("Active %", f"{active_pct:.1f}%")
            with col4:
                st.metric("Total Selected", len(timeline_df))
        else:
            st.info("No repositories available for the selected filters.")

except FileNotFoundError:
    st.error("⚠️ Could not find 'healthcare_data.csv'. Please ensure the file is in the same directory as this script.")
except Exception as e:
    st.error(f"⚠️ An error occurred: {str(e)}")
    st.info("Please check that your data file is properly formatted and all required columns are present.")