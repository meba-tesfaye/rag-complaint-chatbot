import os
import re
import pandas as pd

def run_preprocessing_pipeline(raw_data_path, output_data_path):
    print("🚀 Starting Robust Memory-Efficient Preprocessing Pipeline...")
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}.")
        
    required_cols = ['Complaint ID', 'Product', 'Consumer complaint narrative']
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    
    chunk_size = 50000 
    is_first_chunk = True
    total_processed = 0
    total_saved = 0
    
    print(f"📥 Streaming dataset in batches of {chunk_size:,} rows...")
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        boilerplate_patterns = [
            r"i am writing to file a complaint",
            r"to whom it may concern",
            r"dear consumer financial protection bureau",
            r"xxxx"
        ]
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    # Mapping raw variations to our 4 target standardized categories
    def standardize_product(prod_name):
        if not isinstance(prod_name, str):
            return None
        p = prod_name.lower().strip()
        if 'credit card' in p:
            return 'Credit Card'
        elif 'loan' in p or 'mortgage' in p:
            return 'Personal Loan'
        elif 'savings' in p or 'checking' in p or 'bank account' in p:
            return 'Savings Account'
        elif 'transfer' in p or 'remittance' in p or 'money' in p:
            return 'Money Transfer'
        return None

    for chunk in pd.read_csv(raw_data_path, usecols=required_cols, chunksize=chunk_size, low_memory=False):
        total_processed += len(chunk)
        
        chunk = chunk.dropna(subset=['Product', 'Consumer complaint narrative'])
        
        # Apply standardizer and drop unmapped rows
        chunk['Product_Cleaned'] = chunk['Product'].apply(standardize_product)
        chunk = chunk.dropna(subset=['Product_Cleaned'])
        
        if chunk.empty:
            continue
            
        # Overwrite original product column with the standardized one
        chunk['Product'] = chunk['Product_Cleaned']
        
        # Clean text narratives
        chunk['cleaned_narrative'] = chunk['Consumer complaint narrative'].apply(clean_text)
        chunk = chunk[chunk['cleaned_narrative'].str.len() > 0]
        
        if chunk.empty:
            continue
            
        chunk['narrative_word_count'] = chunk['cleaned_narrative'].apply(lambda x: len(x.split()))
        chunk = chunk.drop(columns=['Product_Cleaned'])
        
        if is_first_chunk:
            chunk.to_csv(output_data_path, mode='w', index=False)
            is_first_chunk = False
        else:
            chunk.to_csv(output_data_path, mode='a', header=False, index=False)
            
        total_saved += len(chunk)
        
    print("\n✅ Robust Preprocessing complete!")
    print(f"📊 Grand Total Rows Examined: {total_processed:,}")
    print(f"💾 Total Extracted & Saved to {output_data_path}: {total_saved:,}")

if __name__ == "__main__":
    RAW_INPUT = "data/raw/complaints.csv" 
    PROCESSED_OUTPUT = "data/processed/filtered_complaints.csv"
    run_preprocessing_pipeline(RAW_INPUT, PROCESSED_OUTPUT)