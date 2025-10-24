import streamlit as st
import pandas as pd
import plotly.express as px
from data_processor import ComplaintDataProcessor
import time

# Configure page
st.set_page_config(
    page_title='Vehicle Complaint Analysis',
    layout='wide',
    initial_sidebar_state='expanded'
)

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
    
    # Add stop button at top
    st.sidebar.button("Stop Processing", key="stop_btn", 
                     on_click=lambda: processor.stop_processing())
    
    # Overview section
    st.header('Dataset Overview')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Total Complaints', len(df))
    
    with col2:
        st.metric('Unique Components', df['component'].nunique())
        
    with col3:
        if not df.empty and 'manufacturer' in df.columns:
            top_mfr = df['manufacturer'].value_counts().index[0] if len(df['manufacturer'].value_counts()) > 0 else 'N/A'
            st.metric('Top Manufacturer', top_mfr)
        else:
            st.metric('Top Manufacturer', 'N/A')
    
    # Add manufacturer filter
    st.sidebar.header("Filters")
    all_manufacturers = ['All'] + sorted(df['manufacturer'].unique().tolist())
    selected_manufacturer = st.sidebar.selectbox(
        'Select Manufacturer', 
        all_manufacturers,
        index=0
    )
    
    # Filter data based on selection
    if selected_manufacturer != 'All':
        filtered_df = df[df['manufacturer'] == selected_manufacturer]
    else:
        filtered_df = df
    
    # Market Basket Analysis Section
    st.header('Component Association Analysis')
    
    with st.expander('Configure MBA Parameters'):
        min_support = st.slider('Minimum Support', 0.01, 0.2, 0.05, 0.01)
        metric = st.selectbox('Association Metric', ['lift', 'confidence', 'support'])
        min_threshold = st.slider('Minimum Threshold', 0.1, 5.0, 1.0, 0.1)
    
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
    
    # Date range selector
    date_col = st.selectbox('Select date column for analysis', 
                          ['date_received', 'date_added', 'fail_date'],
                          help='date_received = Column 17 (LDATE - when NHTSA received complaint)')
    
    if st.button('Analyze Date Patterns'):
        with st.spinner('Analyzing date patterns...'):
            date_analysis = processor.analyze_dates(filtered_df, date_column=date_col)
            
            if date_analysis is None:
                st.warning(f'No valid date data available for {date_col}')
            else:
                # Yearly trends
                st.subheader(f'Yearly Complaint Trends ({date_col})')
                fig = px.bar(
                    date_analysis['by_year'], 
                    labels={'index': 'Year', 'value': 'Complaints'},
                    title=f'Complaints by Year'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly trends
                st.subheader('Monthly Complaint Trends')
                monthly_data = date_analysis['by_month'].reset_index()
                monthly_data.columns = ['Month', 'Complaints']
                monthly_data['Month'] = monthly_data['Month'].astype(str)
                fig = px.line(
                    monthly_data,
                    x='Month',
                    y='Complaints',
                    title='Complaints Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekday patterns
                st.subheader('Complaints by Weekday')
                fig = px.bar(
                    date_analysis['by_weekday'],
                    labels={'index': 'Weekday', 'value': 'Complaints'},
                    title='Day of Week Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Component analysis
    st.header('Component Frequency')
    top_n = st.slider('Number of components to show', 5, 50, 20)
    
    component_counts = filtered_df['component'].value_counts().head(top_n)
    fig = px.bar(
        component_counts,
        orientation='h',
        labels={'index': 'Component', 'value': 'Count'},
        height=600,
        title=f'Components for {selected_manufacturer}'
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
    
    # Manufacturer analysis
    st.header('Manufacturer Analysis')
    manufacturer_counts = df['manufacturer'].value_counts().head(10)
    
    fig = px.pie(
        manufacturer_counts,
        names=manufacturer_counts.index,
        values=manufacturer_counts.values
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
