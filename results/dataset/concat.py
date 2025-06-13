import pandas as pd
import sys

def concat_csv_files(file_1, file_2):
    # Read the CSV files into pandas DataFrames
    df_to = pd.read_csv(file_1)
    df_from = pd.read_csv(file_2)

    # Find the common columns
    common_columns = list(set(df_to.columns) & set(df_from.columns))
    df_to = df_to[common_columns]
    df_from = df_from[common_columns]

    # Check if the columns are the same
    if not all(df_to.columns == df_from.columns):
        raise ValueError("the CSV files do not have the same columns.")

    # Append the rows of df_from to df_to
    df_to = pd.concat([df_to, df_from], ignore_index=True)

    return df_to

if __name__ == "__main__":
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]
    output_file = sys.argv[3]

    df = concat_csv_files(file_1, file_2)
    df.to_csv(output_file, index=False)
