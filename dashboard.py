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
    
    # MBA filters to reduce data size
    st.info('💡 **Performance Tip:** Use filters to reduce processing time and memory usage. Start with manufacturer filter.')
    
    # Manufacturer and component filters
    col_mba1, col_mba2, col_mba3 = st.columns([2, 2, 1])
    with col_mba1:
        all_mfrs = ['All'] + sorted(df['manufacturer'].unique().tolist())
        mba_manufacturer = st.selectbox(
            'Filter by Manufacturer',
            all_mfrs,
            index=0,
            key='mba_mfr_filter',
            help='Filtering by manufacturer significantly reduces memory usage'
        )
    
    with col_mba2:
        # Component filter
        all_components = sorted(df['component'].dropna().unique().tolist())
        selected_components = st.multiselect(
            'Filter by Components (optional)',
            all_components,
            default=[],
            key='mba_comp_filter',
            help='Select specific components to analyze. Leave empty for all components.'
        )
    
    with col_mba3:
        max_transactions = st.number_input(
            'Max Transactions',
            min_value=1000,
            max_value=100000,
            value=50000,
            step=5000,
            help='Limit number of transactions to prevent memory errors'
        )
    
    # Filter data for MBA
    if mba_manufacturer != 'All':
        filtered_df = df[df['manufacturer'] == mba_manufacturer]
    else:
        filtered_df = df
    
    # Apply component filter if selected
    if selected_components:
        filtered_df = filtered_df[filtered_df['component'].isin(selected_components)]
        st.caption(f"🔍 Analyzing {len(selected_components)} selected components")
    
    # Show data size warning
    unique_complaints = filtered_df['cmplid'].nunique()
    st.caption(f"📊 Dataset size: {len(filtered_df):,} records | {unique_complaints:,} unique complaints")
    
    if unique_complaints > 100000:
        st.warning('⚠️ Large dataset detected. Consider filtering by manufacturer or reducing max transactions.')
    
    # Parameter explanation section
    with st.expander('ℹ️ Understanding MBA Parameters', expanded=False):
        st.markdown("""
        ### Parameter Guide
        
        #### 📊 Minimum Support
        **Purpose:** Filters out itemsets that appear too infrequently to be meaningful.
        
        - **Range:** 0.05 – 0.2 (5% to 20%)
        - **Default:** 0.08 (8%) - recommended starting point
        - **Memory-safe:** Higher values (8%+) prevent memory issues
        - **To find more patterns:** Lower to 5-6% if needed
        
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
                min_value=0.05,
                max_value=0.2,
                value=0.08,
                step=0.01,
                format="%.2f",
                help="Filters itemsets by frequency. Higher values prevent memory issues. Start with 8% and adjust."
            )
            st.caption(f"📌 Current: {min_support:.2f} ({min_support*100:.0f}% of transactions)")
        
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
                rules = processor.perform_mba(filtered_df, min_support, metric, min_threshold, max_transactions)
                
                if rules.empty:
                    st.warning('⚠️ No association rules found with the current parameters.')
                    st.info("""
                    **Troubleshooting Tips:**
                    1. **Lower the Minimum Support** to 0.001 or 0.002 to find rarer patterns
                    2. **Select a specific manufacturer** to focus on one brand's data
                    3. **Increase Max Transactions** if you filtered by manufacturer
                    4. **Check if vehicles have multiple component complaints** - MBA requires vehicles with 2+ different component issues
                    
                    The analysis groups complaints by vehicle (manufacturer + make + model + year) and looks for components that fail together.
                    """)
                else:
                    # Summary statistics
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric('Total Rules Found', len(rules))
                    with col_stat2:
                        st.metric('Avg Lift', f"{rules['lift'].mean():.2f}")
                    with col_stat3:
                        st.metric('Avg Confidence', f"{rules['confidence'].mean():.2%}")
                    
                    # Show top rules with column selection
                    st.subheader(f'🏆 Top Component Associations (by {metric})')
                    
                    # Column selector
                    available_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 
                                     'leverage', 'conviction']
                    default_cols = ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
                    
                    # Filter to only show columns that exist in the rules dataframe
                    available_cols = [col for col in available_cols if col in rules.columns]
                    default_cols = [col for col in default_cols if col in rules.columns]
                    
                    with st.expander('⚙️ Customize Table Columns', expanded=False):
                        display_cols = st.multiselect(
                            'Select columns to display',
                            available_cols,
                            default=default_cols,
                            help='Choose which metrics to show in the results table'
                        )
                    
                    if not display_cols:
                        display_cols = default_cols
                    
                    st.dataframe(
                        rules[display_cols].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualizations
                    st.subheader('📊 Association Visualizations')
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4 = st.tabs(['Scatter Plot', 'Top Rules', 'Lift Distribution', 'Network View'])
                    
                    with tab1:
                        # Support vs Confidence scatter
                        fig = px.scatter(
                            rules.head(50),
                            x='support',
                            y='confidence',
                            color='lift',
                            hover_data=['antecedents_str', 'consequents_str'],
                            size='lift',
                            title='Support vs Confidence (colored by Lift)',
                            labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Top rules by lift
                        top_rules = rules.head(15).copy()
                        top_rules['rule'] = top_rules['antecedents_str'] + ' → ' + top_rules['consequents_str']
                        fig = px.bar(
                            top_rules,
                            x='lift',
                            y='rule',
                            orientation='h',
                            title=f'Top 15 Rules by {metric.capitalize()}',
                            labels={'lift': 'Lift', 'rule': 'Association Rule'},
                            color='lift',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Lift distribution
                        fig = px.histogram(
                            rules,
                            x='lift',
                            nbins=30,
                            title='Distribution of Lift Values',
                            labels={'lift': 'Lift', 'count': 'Number of Rules'}
                        )
                        fig.add_vline(x=1, line_dash="dash", line_color="red", 
                                     annotation_text="Lift = 1 (Independence)")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.caption(f"📊 {len(rules[rules['lift'] > 1])} rules with positive association (Lift > 1)")
                    
                    with tab4:
                        # Network visualization using plotly
                        st.markdown("### Component Association Network")
                        st.caption("Shows the top 20 strongest associations")
                        
                        # Prepare network data
                        top_network = rules.head(20)
                        
                        # Create nodes and edges
                        nodes = set()
                        edges = []
                        for _, row in top_network.iterrows():
                            ant = row['antecedents_str']
                            cons = row['consequents_str']
                            nodes.add(ant)
                            nodes.add(cons)
                            edges.append({
                                'source': ant,
                                'target': cons,
                                'lift': row['lift'],
                                'confidence': row['confidence']
                            })
                        
                        # Create a simple network visualization
                        import numpy as np
                        nodes_list = list(nodes)
                        n = len(nodes_list)
                        
                        # Circular layout
                        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
                        x_nodes = np.cos(angles)
                        y_nodes = np.sin(angles)
                        
                        # Create edge traces
                        edge_traces = []
                        for edge in edges:
                            source_idx = nodes_list.index(edge['source'])
                            target_idx = nodes_list.index(edge['target'])
                            
                            edge_trace = {
                                'x': [x_nodes[source_idx], x_nodes[target_idx], None],
                                'y': [y_nodes[source_idx], y_nodes[target_idx], None],
                                'mode': 'lines',
                                'line': {'width': edge['lift'], 'color': 'lightgray'},
                                'hoverinfo': 'text',
                                'text': f"{edge['source']} → {edge['target']}<br>Lift: {edge['lift']:.2f}"
                            }
                            edge_traces.append(edge_trace)
                        
                        # Create node trace
                        node_trace = {
                            'x': x_nodes,
                            'y': y_nodes,
                            'mode': 'markers+text',
                            'marker': {'size': 20, 'color': 'lightblue', 'line': {'width': 2, 'color': 'darkblue'}},
                            'text': nodes_list,
                            'textposition': 'top center',
                            'hoverinfo': 'text'
                        }
                        
                        import plotly.graph_objects as go
                        fig = go.Figure(data=edge_traces + [node_trace])
                        fig.update_layout(
                            showlegend=False,
                            hovermode='closest',
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=600,
                            title='Component Association Network'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Brand-specific insights
                    if mba_manufacturer != 'All':
                        st.subheader(f'🔍 {mba_manufacturer} Brand Insights')
                        
                        # Most common component issues
                        brand_components = filtered_df['component'].value_counts().head(10)
                        
                        col_insight1, col_insight2 = st.columns(2)
                        
                        with col_insight1:
                            st.markdown("**Most Complained Components:**")
                            fig = px.bar(
                                brand_components,
                                orientation='h',
                                title=f'Top 10 Components for {mba_manufacturer}',
                                labels={'value': 'Number of Complaints', 'index': 'Component'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_insight2:
                            st.markdown("**Key Association Patterns:**")
                            if len(rules) > 0:
                                # Find most common antecedents
                                all_antecedents = []
                                for ant in rules['antecedents']:
                                    all_antecedents.extend(list(ant))
                                
                                from collections import Counter
                                ant_counts = Counter(all_antecedents).most_common(5)
                                
                                st.markdown(f"**Components that trigger other failures:**")
                                for comp, count in ant_counts:
                                    st.markdown(f"- **{comp}**: appears in {count} association rules")
                                
                                st.markdown(f"\n**Strongest Association:**")
                                top_rule = rules.iloc[0]
                                st.info(f"**{top_rule['antecedents_str']}** → **{top_rule['consequents_str']}**\n\n"
                                       f"- Lift: {top_rule['lift']:.2f}\n"
                                       f"- Confidence: {top_rule['confidence']:.1%}\n"
                                       f"- Support: {top_rule['support']:.1%}")
                            else:
                                st.info("No significant patterns found.")
                    
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
