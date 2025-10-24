import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class ComplaintDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self._stop_processing = False
        self.columns = {
            'id': 0,
            'complaint_id': 1,
            'manufacturer': 2,
            'make': 3,
            'model': 4,
            'year': 5,
            'crash': 6,
            'fail_date': 7,
            'fire': 8,
            'injured': 9,
            'deaths': 10,
            'component': 11,
            'city': 12,
            'state': 13,
            'vin': 14,
            'date_added': 15,
            'date_received': 16,
            'description': 20
        }
        self.date_formats = {
            'fail_date': '%Y%m%d',
            'date_added': '%Y%m%d',
            'date_received': '%Y%m%d'
        }

    def stop_processing(self):
        """Allow external interruption of processing"""
        self._stop_processing = True

    def process_in_chunks(self, chunk_size=10000):
        """Process data with interrupt check"""
        # Try common encodings for the file
        encodings = ['utf-8', 'windows-1252', 'latin1']
        
        for encoding in encodings:
            try:
                chunks = pd.read_csv(
                    self.file_path,
                    chunksize=chunk_size,
                    header=0,
                    dtype=str,
                    encoding=encoding
                )
                # If we get here, the encoding worked
                processed_data = []
                for chunk in chunks:
                    if self._stop_processing:
                        print("Processing stopped by user request")
                        return pd.concat(processed_data) if processed_data else pd.DataFrame()
                        
                    chunk = self._clean_data(chunk)
                    processed_data.append(chunk)
                    
                return pd.concat(processed_data)
                
            except UnicodeDecodeError:
                continue
                
        raise ValueError(f"Could not decode file with tried encodings: {encodings}")

    def _clean_data(self, chunk):
        """Clean and transform data chunk"""
        # Rename columns
        chunk = chunk.rename(columns={f'Column{i+1}': name 
                                     for name, i in self.columns.items()})
        
        # Convert dates
        for col, fmt in self.date_formats.items():
            chunk[col] = pd.to_datetime(chunk[col], format=fmt, errors='coerce')
            
        # Clean components
        chunk['component'] = chunk['component'].str.strip().str.upper()
        
        return chunk

    def prepare_mba_data(self, df):
        """Prepare data for Market Basket Analysis"""
        # Group by vehicle and collect components
        transactions = df.groupby(['manufacturer', 'make', 'model'])['component'] \
                        .apply(list).tolist()
        
        # One-hot encode
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        return pd.DataFrame(te_ary, columns=te.columns_)

    def perform_eda(self, df):
        """Perform exploratory data analysis"""
        eda_results = {
            'component_counts': pd.Series(dtype='int64'),
            'manufacturer_counts': pd.Series(dtype='int64')
        }
        
        if not df.empty:
            eda_results['component_counts'] = df['component'].value_counts()
            eda_results['manufacturer_counts'] = df['manufacturer'].value_counts()
            
        return eda_results

    def visualize_eda(self, eda_results, output_dir='eda_plots'):
        """Generate visualizations with robust empty data handling"""
        os.makedirs(output_dir, exist_ok=True)
        plots_generated = 0
        
        # Component Distribution
        if not eda_results['component_counts'].empty:
            plt.figure(figsize=(12, 8))
            eda_results['component_counts'].head(20).sort_values().plot(kind='barh')
            plt.title('Top 20 Complaint Components')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/component_distribution.png')
            plt.close()
            plots_generated += 1
        
        # Manufacturer Distribution
        if len(eda_results['manufacturer_counts']) > 0:
            plt.figure(figsize=(12, 6))
            eda_results['manufacturer_counts'].head(10).plot(kind='bar')
            plt.title('Top 10 Manufacturers by Complaints')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/manufacturer_distribution.png')
            plt.close()
            plots_generated += 1
        
        print(f"Generated {plots_generated} visualizations in {output_dir}")

    def save_processed_data(self, df, output_path):
        """Save processed data to parquet format"""
        # Ensure all string columns are properly encoded
        str_cols = df.select_dtypes(['object']).columns
        df[str_cols] = df[str_cols].astype(str)
        
        # Save to parquet
        df.to_parquet(output_path, engine='pyarrow')

    def perform_mba(self, df, min_support=0.05, metric='lift', min_threshold=1):
        """Perform Market Basket Analysis on components"""
        # Prepare transaction data
        mba_data = self.prepare_mba_data(df)
        
        # Find frequent itemsets
        frequent_itemsets = apriori(mba_data, min_support=min_support, use_colnames=True)
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        # Add interpretation columns
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        return rules.sort_values(metric, ascending=False)

    def visualize_mba(self, rules, output_dir='mba_results'):
        """Generate MBA visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Support vs Confidence
        plt.figure(figsize=(10, 6))
        plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Support vs Confidence')
        plt.savefig(f'{output_dir}/support_vs_confidence.png')
        plt.close()
        
        # Lift Distribution
        plt.figure(figsize=(10, 6))
        rules['lift'].plot(kind='hist', bins=20)
        plt.title('Lift Distribution')
        plt.savefig(f'{output_dir}/lift_distribution.png')
        plt.close()
