import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder

def process_data(file_path):
    # Process data in chunks
    chunks = pd.read_csv(file_path, sep='\t', chunksize=10000,
                        usecols=[2, 4, 10],  # Company, Model, Component
                        names=['company', 'model', 'component'],
                        header=None)
    
    transactions = []
    for chunk in chunks:
        # Group by complaint ID (assuming first column is ID) and collect components
        grouped = chunk.groupby(['company', 'model'])['component'].apply(list)
        transactions.extend(grouped.values.tolist())
    
    # Convert to one-hot encoded format
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

def main():
    st.title('Market Basket Analysis - Car Complaints')
    
    # Load and process data
    with st.spinner('Processing complaint data...'):
        data = process_data('COMPLAINTS_RECEIVED_2025-2025.txt')
    
    # Apriori algorithm
    frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    
    # Display results
    st.subheader('Association Rules')
    st.dataframe(rules.sort_values('lift', ascending=False))
    
    # Visualizations
    st.subheader('Support vs Confidence')
    fig = px.scatter(rules, x='support', y='confidence', color='lift',
                     hover_data=['antecedents', 'consequents'],
                     size='lift',
                     labels={'support':'Support', 'confidence':'Confidence'})
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
