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


# Function to drop specified columns from a DataFrame
def drop_and_save_columns(df, cols):
    # If a single column name is given as a string, drop it using axis=1
    if isinstance(cols, str):
        return df.drop(cols, axis=1)
    # If a list of column names is given, drop all specified columns
    elif isinstance(cols, list):
        return df.drop(columns=cols)
    # If the input is neither a string nor a list, raise an error
    else:
        raise TypeError("`cols` must be a string or a list of strings")


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