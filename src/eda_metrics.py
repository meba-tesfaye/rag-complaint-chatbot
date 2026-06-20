import os
import pandas as pd

def run_eda_counts(raw_data_path):
    print("📊 Calculating explicit Task 1 EDA metrics...")
    if not os.path.exists(raw_data_path):
        print("Raw file not found.")
        return

    required_cols = ['Consumer complaint narrative', 'Product']
    chunk_size = 100000
    
    total_with_narrative = 0
    total_without_narrative = 0
    short_entries = 0  # less than 5 words
    long_entries = 0   # more than 500 words
    
    for chunk in pd.read_csv(raw_data_path, usecols=required_cols, chunksize=chunk_size, low_memory=False):
        # Count narratives
        has_narrative = chunk['Consumer complaint narrative'].notna()
        total_with_narrative += has_narrative.sum()
        total_without_narrative += (~has_narrative).sum()
        
        # Calculate lengths on non-empty rows
        valid_narratives = chunk['Consumer complaint narrative'].dropna().astype(str)
        word_counts = valid_narratives.apply(lambda x: len(x.split()))
        
        short_entries += (word_counts <= 5).sum()
        long_entries += (word_counts > 500).sum()

    print("\n📈 --- Final EDA Report Metrics ---")
    print(f"Complaints WITH Narratives: {total_with_narrative:,}")
    print(f"Complaints WITHOUT Narratives: {total_without_narrative:,}")
    print(f"Total Raw Complaints Examined: {(total_with_narrative + total_without_narrative):,}")
    print(f"Very Short Narratives (<= 5 words): {short_entries:,}")
    print(f"Very Long Narratives (> 500 words): {long_entries:,}")

if __name__ == "__main__":
    run_eda_counts("data/raw/complaints.csv")