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
        st.metric('Top Manufacturer', df['manufacturer'].value_counts().index[0])
    
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
    
    if st.button('Run Market Basket Analysis'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Process with progress updates
            for i in range(100):
                if processor._stop_processing:
                    status_text.warning("Processing stopped")
                    break
                
                progress_bar.progress(i + 1)
                time.sleep(0.1)  # Simulate processing
                
            if not processor._stop_processing:
                rules = processor.perform_mba(filtered_df, min_support, metric, min_threshold)
                
                # Show top rules
                st.subheader(f'Top Component Associations (by {metric})')
                st.dataframe(rules.head(20))
                
                # Visualizations
                st.subheader('Association Visualizations')
                
                fig = px.scatter(
                    rules,
                    x='support',
                    y='confidence',
                    color='lift',
                    hover_data=['antecedents_str', 'consequents_str'],
                    size='lift'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            processor._stop_processing = False
    
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
    
    # Add click interaction
    if st.session_state.get('click_data'):
        clicked_component = st.session_state.click_data['points'][0]['y']
        st.write(f"Showing details for: {clicked_component}")
        st.dataframe(
            filtered_df[filtered_df['component'] == clicked_component]\
                [['manufacturer', 'make', 'model', 'description']]\
                .head(20)
        )
    
    # JavaScript to capture clicks
    st.components.v1.html("""
    <script>
    window.onload = function() {
        const iframe = parent.document.querySelector("iframe[data-testid='stPlotlyChart']");
        iframe.contentWindow.document.querySelector('.plotly').on('plotly_click', 
            function(data) {
                parent.window.stSessionState.set('click_data', data);
                parent.window.stSessionState.rerun();
            }
        );
    }
    </script>
    """, height=0)
    
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
