# project.py

import pandas as pd
import sys
import ast

def project_dataframe(input_file, columns_to_keep):
    """Projects a CSV table, returning a pandas DataFrame with specified columns.

    Args:
        input_file: The path to the input CSV file.
        columns_to_keep: A list of column titles to keep.
    
    Returns:
        pandas.DataFrame: A DataFrame containing only the specified columns.
    """
    
    # Read the CSV directly into a DataFrame
    df = pd.read_csv(input_file)
    
    # Check if all columns_to_keep exist in the DataFrame
    valid_columns = [col for col in columns_to_keep if col in df.columns]
    
    # Return the projected DataFrame
    return df[valid_columns]

if __name__ == "__main__":
    # Get user input (or modify this section for direct file paths)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    columns_to_keep = ast.literal_eval(sys.argv[3])

    projected_df = project_dataframe(input_file, columns_to_keep)
    projected_df.to_csv(output_file, index=False)