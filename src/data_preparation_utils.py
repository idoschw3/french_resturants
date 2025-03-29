import pandas as pd
import re

# Function to load data based on file extension
def load_data(filepath):
    # If the file is a CSV, use pandas to read it
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    # If the file is an Excel file, use pandas to read it
    elif filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    # If the file format is unsupported, raise an error
    else:
        raise ValueError("Unsupported file type")


# Function to filter the dataframe by a specific value in a given column
def filter_rows_by_column_value(df, column, value, clean_func=None):
    # If a cleaning function is provided, apply it to the column and the value
    if clean_func:
        df[column] = df[column].apply(clean_func)
        value = clean_func(value)

    # Return only the rows where the column matches the value
    return df[df[column] == value].copy()


# Function to drop specified columns from a DataFrame and store them separately
def drop_and_save_columns(df, cols, dropped_df=None):
    """
    Drops columns from the given DataFrame and appends them to a separate DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame.
        cols (str or list): Column name or list of column names to drop.
        dropped_df (pd.DataFrame, optional): A separate DataFrame to collect dropped columns.

    Returns:
        tuple: (df_after_dropping, dropped_columns_df)
    """
    # Normalize cols to list
    if isinstance(cols, str):
        cols = [cols]
    elif not isinstance(cols, list):
        raise TypeError("`cols` must be a string or a list of strings")

    # Extract dropped columns
    dropped = df[cols].copy()

    # Drop from original DataFrame
    df = df.drop(columns=cols)

    # Append dropped columns to dropped_df
    if dropped_df is not None:
        dropped_df = pd.concat([dropped_df, dropped], axis=1)
    else:
        dropped_df = dropped

    return df, dropped_df


# Function to clean and standardize text entries
def clean_text(text):
    # If the input is not a string (e.g., NaN), return it unchanged
    if not isinstance(text, str):
        return text
    # Convert to lowercase and remove leading/trailing whitespace
    text = text.strip().lower()
    # Replace hyphens with spaces to separate joined words
    text = re.sub(r'[-]', ' ', text)
    # Add a space between lowercase-uppercase pairs (e.g., "northWest" → "north West")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove all characters that are not lowercase letters, commas, or spaces
    text = re.sub(r'[^a-z\s,]', '', text)
    # Final strip to clean any remaining leading/trailing spaces
    return text.strip()