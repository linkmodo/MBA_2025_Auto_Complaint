from data_processor import ComplaintDataProcessor
import os

def main():
    processor = ComplaintDataProcessor('COMPLAINTS_RECEIVED_2025-2025.txt')
    
    # Process data in chunks
    print("Processing data in chunks...")
    processed_data = processor.process_in_chunks()
    
    # Perform EDA
    print("Performing EDA...")
    eda_results = processor.perform_eda(processed_data)
    
    # Save processed data
    os.makedirs('processed_data', exist_ok=True)
    processor.save_processed_data(processed_data, 'processed_data/complaints_2025.parquet')
    
    # Generate visualizations
    os.makedirs('eda_plots', exist_ok=True)
    processor.visualize_eda(eda_results)
    
    print("Data processing complete!")

if __name__ == '__main__':
    main()
