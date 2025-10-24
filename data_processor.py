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
        # Column mapping based on column description.txt (1-indexed columns)
        # CSV columns are named Column1, Column2, etc.
        self.column_names = [
            'cmplid',           # Column1
            'odino',            # Column2
            'manufacturer',     # Column3
            'make',             # Column4
            'model',            # Column5
            'year',             # Column6
            'crash',            # Column7
            'fail_date',        # Column8
            'fire',             # Column9
            'injured',          # Column10
            'deaths',           # Column11
            'component',        # Column12
            'city',             # Column13
            'state',            # Column14
            'vin',              # Column15
            'date_added',       # Column16 - DATEA
            'date_received',    # Column17 - LDATE (complaint received by NHTSA)
            'miles',            # Column18
            'occurrences',      # Column19
            'description'       # Column20 - CDESCR
        ]
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
        # Rename columns - CSV has Column1, Column2, etc.
        rename_dict = {f'Column{i+1}': name for i, name in enumerate(self.column_names)}
        chunk = chunk.rename(columns=rename_dict)
        
        # Convert dates
        for col, fmt in self.date_formats.items():
            chunk[col] = pd.to_datetime(chunk[col], format=fmt, errors='coerce')
            
        # Clean components
        chunk['component'] = chunk['component'].str.strip().str.upper()
        
        return chunk

    def prepare_mba_data(self, df, max_transactions=50000):
        """Prepare data for Market Basket Analysis
        
        Args:
            df: DataFrame with complaint data
            max_transactions: Maximum number of transactions to process (default 50000)
        """
        # Group by vehicle (manufacturer + make + model + year) to find component associations
        # Each unique vehicle becomes a "transaction" with its associated complaint components
        transactions = df.groupby(['manufacturer', 'make', 'model', 'year'])['component'] \
                        .apply(lambda x: list(set(x))).tolist()  # Use set to get unique components per vehicle
        
        # Filter out transactions with only 1 item (no associations possible)
        transactions = [t for t in transactions if len(t) > 1]
        
        # Limit transactions to prevent memory issues
        if len(transactions) > max_transactions:
            print(f"Warning: Limiting to {max_transactions} transactions (from {len(transactions)})")
            transactions = transactions[:max_transactions]
        
        if len(transactions) == 0:
            print("Warning: No transactions with multiple components found")
            return pd.DataFrame()
        
        print(f"Processing {len(transactions)} transactions with multiple components")
        
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

    def perform_mba(self, df, min_support=0.05, metric='lift', min_threshold=1, max_transactions=50000):
        """Perform Market Basket Analysis on components
        
        Args:
            df: DataFrame with complaint data
            min_support: Minimum support threshold
            metric: Metric to use for rules (lift, confidence, support)
            min_threshold: Minimum threshold for the metric
            max_transactions: Maximum transactions to process
        """
        # Prepare transaction data
        mba_data = self.prepare_mba_data(df, max_transactions=max_transactions)
        
        if mba_data.empty:
            return pd.DataFrame()
        
        # Find frequent itemsets
        frequent_itemsets = apriori(mba_data, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        if rules.empty:
            return pd.DataFrame()
        
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

    def analyze_dates(self, df, date_column='date_received'):
        """Analyze date patterns in complaint data
        
        Args:
            df: DataFrame with complaint data
            date_column: Column to analyze ('date_received', 'date_added', or 'fail_date')
        """
        if date_column not in df.columns or df[date_column].isna().all():
            return None
            
        date_analysis = {
            'by_year': df[date_column].dt.year.value_counts().sort_index(),
            'by_month': df[date_column].dt.to_period('M').value_counts().sort_index(),
            'by_weekday': df[date_column].dt.day_name().value_counts()
        }
        return date_analysis

    def visualize_date_analysis(self, date_analysis, output_dir='date_plots'):
        """Generate date analysis visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        plots_generated = 0
        
        # Yearly trends
        plt.figure(figsize=(12, 6))
        date_analysis['by_year'].plot(kind='bar')
        plt.title('Complaints by Year')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/complaints_by_year.png')
        plt.close()
        plots_generated += 1
        
        # Monthly trends
        plt.figure(figsize=(12, 6))
        date_analysis['by_month'].plot(kind='line')
        plt.title('Complaints by Month')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/complaints_by_month.png')
        plt.close()
        plots_generated += 1
        
        print(f"Generated {plots_generated} date analysis visualizations in {output_dir}")
