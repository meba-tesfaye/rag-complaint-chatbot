import os
import pandas as pd

def run_eda_counts(raw_data_path):
    print("📊 Calculating explicit Task 1 EDA metrics...")
    if not os.path.exists(raw_data_path):
        print("Raw file not found.")
        return
    
    required_cols = ['Consumer complaint narrative']
    chunk_size = 100000
    total_with_narrative = 0
    total_without_narrative = 0
    
    for chunk in pd.read_csv(raw_data_path, usecols=required_cols, chunksize=chunk_size, low_memory=False):
        has_narrative = chunk['Consumer complaint narrative'].notna()
        total_with_narrative += has_narrative.sum()
        total_without_narrative += (~has_narrative).sum()
        
    print("\n📈 --- Final EDA Report Metrics ---")
    print(f"Complaints WITH Narratives: {total_with_narrative:,}")
    print(f"Complaints WITHOUT Narratives: {total_without_narrative:,}")
    print(f"Total Raw Complaints Examined: {(total_with_narrative + total_without_narrative):,}")

if __name__ == '__main__':
    run_eda_counts("data/raw/complaints.csv")
