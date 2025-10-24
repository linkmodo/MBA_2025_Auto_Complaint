import streamlit as st
import pandas as pd
import plotly.express as px
from data_processor import ComplaintDataProcessor
import time

# Configure page
st.set_page_config(
    page_title='Vehicle Complaint Analysis',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Custom CSS for layout
st.markdown("""
<style>
    .main .block-container {
        max-width: 950px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Reduce dataset overview font size */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    processor = ComplaintDataProcessor('Vehicle Complaint Data.csv')
    df = processor.process_in_chunks()
    return df, processor

def main():
    st.title('🚗 Vehicle Complaint Analysis Dashboard')
    
    # Load data with progress indicator
    with st.spinner('Loading complaint data...'):
        df, processor = load_data()
    
    # Overview section
    st.header('Dataset Overview')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Total Complaints', len(df))
    
    with col2:
        st.metric('Unique Components', df['component'].nunique())
        
    with col3:
        if not df.empty and 'manufacturer' in df.columns:
            mfr_count = df['manufacturer'].nunique()
            st.metric('Number of Manufacturers', mfr_count)
        else:
            st.metric('Number of Manufacturers', 0)
    
    # Market Basket Analysis Section
    st.header('🔍 Component Association Analysis')
    
    # MBA uses full dataset (filtering happens in Component Frequency section)
    filtered_df = df
    
    # Parameter explanation section
    with st.expander('ℹ️ Understanding MBA Parameters', expanded=False):
        st.markdown("""
        ### Parameter Guide
        
        #### 📊 Minimum Support
        **Purpose:** Filters out itemsets that appear too infrequently to be meaningful.
        
        - **Typical Range:** 0.001 – 0.1 (0.1% to 10%)
        - **Default:** 0.01 (1%) - recommended starting point
        - **Large datasets:** Use smaller support (e.g., 0.001)
        - **Smaller datasets:** Use higher support (e.g., 0.05–0.1)
        
        #### 📈 Association Metric
        Used to rank or filter association rules once frequent itemsets are found.
        
        | Metric | Description | When to Use |
        |--------|-------------|-------------|
        | **Support** | How frequently the rule occurs | High statistical reliability needed; common in retail |
        | **Confidence** | How often Y occurs when X occurs (P(Y given X)) | Predictive relationships; understand conditional probability |
        | **Lift** | How much more likely Y is given X vs. random | **Most common** - identifies interesting non-trivial relationships; corrects popularity bias |
        
        #### 🎯 Minimum Threshold
        Sets the cutoff value for the chosen metric.
        
        - **For Support:** 0.001 – 0.05 (default ~0.01)
        - **For Confidence:** 0.3 – 0.8 (default ~0.5)
        - **For Lift:** Usually >1.0 (default = 1) — rules with lift > 1 are positively correlated
        
        ---
        
        **💡 Recommended Starting Configuration:**
        - Minimum Support: **0.01**
        - Metric: **lift**
        - Minimum Threshold: **1.0**
        """)
    
    with st.expander('⚙️ Configure MBA Parameters', expanded=True):
        st.info('💡 **Tip:** Start with default values, then adjust based on results. Lower support finds rarer patterns; higher lift finds stronger associations.')
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_support = st.slider(
                'Minimum Support',
                min_value=0.001,
                max_value=0.2,
                value=0.01,
                step=0.001,
                format="%.3f",
                help="Filters itemsets by frequency. Lower = find rarer patterns. Higher = only common patterns."
            )
            st.caption(f"📌 Current: {min_support:.3f} ({min_support*100:.1f}% of transactions)")
        
        with col2:
            # Dynamic threshold based on metric
            metric = st.selectbox(
                'Association Metric',
                options=['lift', 'confidence', 'support'],
                index=0,
                help="**Lift** (recommended): Measures strength vs. random chance\n**Confidence**: Predictive accuracy\n**Support**: Frequency of occurrence"
            )
            
            # Set threshold ranges based on metric
            if metric == 'lift':
                min_threshold = st.slider(
                    'Minimum Threshold (Lift)',
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    help="Lift > 1 = positive correlation, = 1 independent, < 1 negative correlation"
                )
                st.caption(f"📌 Current: {min_threshold:.1f} ({'Positive' if min_threshold > 1 else 'Negative' if min_threshold < 1 else 'Independent'} association)")
            elif metric == 'confidence':
                min_threshold = st.slider(
                    'Minimum Threshold (Confidence)',
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Probability that Y occurs given X. Higher = stronger prediction."
                )
                st.caption(f"📌 Current: {min_threshold:.2f} ({min_threshold*100:.0f}% confidence)")
            else:  # support
                min_threshold = st.slider(
                    'Minimum Threshold (Support)',
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Minimum frequency for the complete rule (X→Y together)"
                )
                st.caption(f"📌 Current: {min_threshold:.3f} ({min_threshold*100:.1f}% of transactions)")
    
    if st.button('Run Market Basket Analysis', type='primary'):
        with st.spinner('Running Market Basket Analysis...'):
            try:
                rules = processor.perform_mba(filtered_df, min_support, metric, min_threshold)
                
                if rules.empty:
                    st.warning('No association rules found with the current parameters. Try lowering the thresholds.')
                else:
                    # Show top rules
                    st.subheader(f'Top Component Associations (by {metric})')
                    display_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
                    st.dataframe(
                        rules[display_cols].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualizations
                    st.subheader('Association Visualizations')
                    
                    fig = px.scatter(
                        rules,
                        x='support',
                        y='confidence',
                        color='lift',
                        hover_data=['antecedents_str', 'consequents_str'],
                        size='lift',
                        title='Support vs Confidence (colored by Lift)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error during MBA: {str(e)}")
                st.info('Try adjusting the parameters or selecting a specific manufacturer to reduce data size.')
    
    # Date Analysis Section
    st.header('📅 Complaint Date Analysis')
    
    col_date1, col_date2 = st.columns([2, 1])
    with col_date1:
        date_col = st.selectbox('Select date column for analysis', 
                              ['date_received', 'date_added', 'fail_date'],
                              help='date_received = Column 17 (LDATE - when NHTSA received complaint)')
    with col_date2:
        view_type = st.selectbox('View by', ['Month', 'Year', 'Day of Week'])
    
    if st.button('Analyze Date Patterns'):
        with st.spinner('Analyzing date patterns...'):
            date_analysis = processor.analyze_dates(filtered_df, date_column=date_col)
            
            if date_analysis is None:
                st.warning(f'No valid date data available for {date_col}')
            else:
                # Display based on selected view type
                if view_type == 'Month':
                    st.subheader(f'Monthly Complaint Trends ({date_col})')
                    monthly_data = date_analysis['by_month'].reset_index()
                    monthly_data.columns = ['Month', 'Complaints']
                    monthly_data['Month'] = monthly_data['Month'].astype(str)
                    fig = px.line(
                        monthly_data,
                        x='Month',
                        y='Complaints',
                        title='Complaints Over Time',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif view_type == 'Year':
                    st.subheader(f'Yearly Complaint Trends ({date_col})')
                    fig = px.bar(
                        date_analysis['by_year'], 
                        labels={'index': 'Year', 'value': 'Complaints'},
                        title=f'Complaints by Year'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # Day of Week
                    st.subheader(f'Complaints by Weekday ({date_col})')
                    fig = px.bar(
                        date_analysis['by_weekday'],
                        labels={'index': 'Weekday', 'value': 'Complaints'},
                        title='Day of Week Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Manufacturer analysis (moved before component analysis)
    st.header('Manufacturer Analysis')
    manufacturer_counts = df['manufacturer'].value_counts().head(10)
    
    fig_mfr = px.pie(
        manufacturer_counts,
        names=manufacturer_counts.index,
        values=manufacturer_counts.values,
        title='Top 10 Manufacturers by Complaint Count'
    )
    fig_mfr.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_mfr, use_container_width=True)
    
    # Store selected manufacturer from pie chart click
    if 'selected_mfr' not in st.session_state:
        st.session_state.selected_mfr = 'All'
    
    # Manual manufacturer selector
    all_manufacturers = ['All'] + sorted(df['manufacturer'].unique().tolist())
    selected_manufacturer = st.selectbox(
        '🔍 Select manufacturer to view components:',
        all_manufacturers,
        index=all_manufacturers.index(st.session_state.selected_mfr) if st.session_state.selected_mfr in all_manufacturers else 0,
        key='mfr_selector'
    )
    st.session_state.selected_mfr = selected_manufacturer
    
    # Component analysis
    st.header('Component Frequency')
    
    top_n = st.slider('Number of components to show', 5, 50, 20)
    
    # Filter data based on selection
    if selected_manufacturer != 'All':
        filtered_df = df[df['manufacturer'] == selected_manufacturer]
    else:
        filtered_df = df
    
    component_counts = filtered_df['component'].value_counts().head(top_n)
    fig = px.bar(
        component_counts,
        orientation='h',
        labels={'index': 'Component', 'value': 'Count'},
        height=600,
        title=f'Top {top_n} Components for {selected_manufacturer}'
    )
    
    # Make bars clickable for drill-down
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
        customdata=component_counts.index
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Component details section
    with st.expander('View Component Details'):
        selected_component = st.selectbox(
            'Select a component to view details',
            options=component_counts.index.tolist()
        )
        if selected_component:
            component_details = filtered_df[filtered_df['component'] == selected_component]\
                [['manufacturer', 'make', 'model', 'year', 'state', 'description']]\
                .head(20)
            st.dataframe(component_details, use_container_width=True)
    

if __name__ == '__main__':
    main()
