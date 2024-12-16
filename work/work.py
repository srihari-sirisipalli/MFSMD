import pandas as pd

def reshape_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Calculate number of groups (each group has 8 rows)
    n_groups = len(df) // 8
    
    reshaped_data = []
    
    # Process each group of 8 rows
    for i in range(n_groups):
        group = df.iloc[i*8:(i+1)*8]
        row_dict = {}
        
        # Process each row in the group
        for j, row in enumerate(group.itertuples(), 1):
            prefix = f"{j}_"
            
            # Get all columns except the index and filename/channel
            metrics = row._asdict()
            for col in df.columns:
                if col not in ['Index', 'file_name', 'channel']:
                    row_dict[prefix + col] = metrics[col]
        
        reshaped_data.append(row_dict)
    
    # Create final dataframe
    result_df = pd.DataFrame(reshaped_data)
    
    # Sort columns to ensure consistent order
    result_df = result_df.reindex(sorted(result_df.columns), axis=1)
    
    # Save to new CSV file
    result_df.to_csv(output_file, index=False)
    
    return result_df

# Example usage:
input_file = r"C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\MFDMS\unbalance_time_domain_features.csv"
output_file = 'unbalance_time_domain_features_reshaped.csv'  # Replace with desired output filename

df = reshape_csv(input_file, output_file)

# Print summary of the transformation
print(f"Transformation complete!")
print(f"Original shape: {len(df)*8} rows x {len(df.columns)//8} columns")
print(f"New shape: {len(df)} rows x {len(df.columns)} columns")
print("\nFirst few columns:")
print(df)