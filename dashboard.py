import streamlit as st
import pandas as pd
import plotly.express as px
from data_processor import ComplaintDataProcessor
import time
import json
from pathlib import Path

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
    
    # Enhanced Filters Section
    st.sidebar.header("Advanced Filters")
    
    # 1. Manufacturer Multi-Select
    manufacturers = processor.get_unique_values(df, 'manufacturer')
    selected_mfgs = st.sidebar.multiselect(
        'Manufacturer(s)', 
        manufacturers[1:],  # Exclude 'All'
        default=manufacturers[1] if len(manufacturers) > 1 else None
    )
    
    # 2. Model Year Range
    valid_years = [y for y in processor.get_unique_values(df, 'year') if y != 'All' and int(y) <= 2026]
    if len(valid_years) > 1:
        year_range = st.sidebar.slider(
            'Model Year Range',
            min_value=int(valid_years[0]),
            max_value=min(2026, int(valid_years[-1])),  
            value=(int(valid_years[0]), min(2026, int(valid_years[-1])))
        )
    
    # 3. Date Range Filter (using fail_date)
    if 'fail_date' in df.columns:
        min_date = df['fail_date'].min()
        max_date = df['fail_date'].max()
        date_range = st.sidebar.date_input(
            'Incident Date Range',
            value=(min_date, max_date)
        )
    
    # 4. Component Type Filter
    components = processor.get_unique_values(df, 'component')
    selected_components = st.sidebar.multiselect(
        'Component(s)',
        components[1:],
        default=None
    )
    
    # Filter Presets Management
    def get_current_filters():
        """Return current filter states as dict"""
        return {
            'manufacturers': st.session_state.get('selected_mfgs', []),
            'year_range': st.session_state.get('year_range', (2010, 2026)),
            'date_range': st.session_state.get('date_range', None),
            'components': st.session_state.get('selected_components', [])
        }

    def save_filter_preset(name):
        """Save current filters as named preset"""
        presets_dir = Path('filter_presets')
        presets_dir.mkdir(exist_ok=True)
        
        preset_path = presets_dir / f'{name.lower().replace(" ", "_")}.json'
        with open(preset_path, 'w') as f:
            json.dump(get_current_filters(), f)

    def load_filter_presets():
        """Load all saved presets"""
        presets = {}
        presets_dir = Path('filter_presets')
        
        if presets_dir.exists():
            for preset_file in presets_dir.glob('*.json'):
                with open(preset_file) as f:
                    presets[preset_file.stem.replace('_', ' ').title()] = json.load(f)
        return presets

    # Add to sidebar UI
    def render_preset_ui():
        """Add preset controls to sidebar"""
        st.sidebar.header("Filter Presets")
        presets = load_filter_presets()
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if presets:
                selected_preset = st.selectbox(
                    'Load Preset', 
                    ['Select preset...'] + list(presets.keys())
                )
                if selected_preset != 'Select preset...':
                    return presets[selected_preset]
        
        with col2:
            if st.button('💾 Save Preset'):
                preset_name = st.text_input('Preset Name:', key='preset_name')
                if preset_name:
                    save_filter_preset(preset_name)
                    st.success(f'Saved "{preset_name}"')
        
        return None

    preset_filters = render_preset_ui()
    if preset_filters:
        st.session_state.selected_mfgs = preset_filters['manufacturers']
        st.session_state.year_range = preset_filters['year_range']
        st.session_state.date_range = preset_filters['date_range']
        st.session_state.selected_components = preset_filters['components']

    # Apply Filters
    filtered_df = df.copy()
    
    # Manufacturer filter
    if selected_mfgs:
        filtered_df = filtered_df[filtered_df['manufacturer'].isin(selected_mfgs)]
    
    # Year filter
    if 'year_range' in locals():
        filtered_df = filtered_df[
            (filtered_df['year'].astype(int) >= year_range[0]) & 
            (filtered_df['year'].astype(int) <= year_range[1])
        ]
    
    # Date filter
    if 'date_range' in locals() and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['fail_date'] >= pd.to_datetime(date_range[0])) & 
            (filtered_df['fail_date'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Component filter
    if selected_components:
        filtered_df = filtered_df[filtered_df['component'].isin(selected_components)]
    
    # Export Filtered Data
    st.sidebar.download_button(
        "Export Filtered Data",
        filtered_df.to_csv(index=False).encode('utf-8'),
        "filtered_complaints.csv",
        "text/csv"
    )
    
    # Overview section
    st.header('Dataset Overview')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Total Complaints', len(filtered_df))
    
    with col2:
        st.metric('Unique Components', filtered_df['component'].nunique())
        
    with col3:
        st.metric('Top Manufacturer', filtered_df['manufacturer'].value_counts().index[0])
    
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
    
    # Visualization-Specific Filters
    with st.expander("🔧 Chart Filters", expanded=False):
        st.markdown("### Component Analysis")
        min_complaints = st.number_input(
            "Minimum complaints threshold",
            min_value=1,
            value=5,
            key='min_complaints'
        )
        
        st.markdown("### Manufacturer Analysis")
        min_mfg_to_show = st.number_input(
            "Minimum complaints to include manufacturer",
            min_value=1,
            value=10,
            key='min_mfg_complaints'
        )
    
    # Component analysis
    st.header('Component Frequency')
    top_n = st.slider('Number of components to show', 5, 50, 20)
    
    component_counts = filtered_df['component'].value_counts()
    component_counts = component_counts[component_counts >= st.session_state.min_complaints].head(top_n)
    fig = px.bar(
        component_counts,
        orientation='h',
        labels={'index': 'Component', 'value': 'Count'},
        height=600,
        title=f'Components for {", ".join(selected_mfgs) if selected_mfgs else "All Manufacturers"}'
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
    mfg_counts = filtered_df['manufacturer'].value_counts()
    mfg_counts = mfg_counts[mfg_counts >= st.session_state.min_mfg_complaints].head(10)
    
    fig = px.pie(
        mfg_counts,
        names=mfg_counts.index,
        values=mfg_counts.values
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
