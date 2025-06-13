# left_join.py

import pandas as pd
import sys


def left_join_csv(file1, file2, left_col, right_col, multiple_map_col):
    """Performs a left join on two CSV files with specified column mappings.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        left_col (str): Column name from file1 to use in the join.
        right_col (str): Corresponding column name from file2 to use in the join.

    Returns:
        pandas.DataFrame: The resulting DataFrame after the join.
    """
    
    # Load data from CSV files into Pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    def aggregate_columns(group):
        """Aggregates columns from the right table, using a list or single value."""
        result = {}
        for col in df2.columns:
            if col != right_col and col != multiple_map_col:  # Exclude the 'SRX' column used for joining
                values = group[col].unique()
                assert len(values) == 1
                result[col] = values[0]
            elif col == multiple_map_col:
                values = group[col].unique()
                result[col] = list(values)
        return pd.Series(result)

    # Perform the inner join 
    joined_df = pd.merge(df1, df2, left_on=left_col, right_on=right_col, how='left')

    grouped_df = joined_df.groupby(left_col).apply(aggregate_columns, include_groups=False).reset_index()

    final_df = pd.merge(df1, grouped_df, on=left_col, how="left")

    return final_df

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # Define corresponding column names for the join
    left_col = sys.argv[3]
    right_col = sys.argv[4]
    multiple_map_col = sys.argv[5]

    result_df = left_join_csv(file1, file2, left_col, right_col, multiple_map_col)

    result_df.to_csv(sys.argv[6], index=False)
