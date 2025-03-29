import pandas as pd
import re
import os

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


# Function to drop one or more columns from a DataFrame, and raise error if any don't exist
def drop_and_save_columns(df, cols, dropped_df=None, save_path=None):

    # Set default save path if none provided
    if save_path is None:
        save_path = r"C:\Users\idosc\Documents\GitHub\french_resturants\data\processed\dropped_columns.csv"

    # Normalize input to a list of column names
    if isinstance(cols, str):
        cols = [cols]
    elif isinstance(cols, list):
        if not all(isinstance(col, str) for col in cols):
            raise TypeError("All column names must be strings")
    else:
        raise TypeError("`cols` must be a string or a list of strings")

    # Check that all columns exist in the DataFrame
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"The following columns were not found in the DataFrame: {missing}")

    # Extract dropped columns
    dropped = df[cols].copy()

    # Drop them from the original DataFrame
    df = df.drop(columns=cols)

    # Append to dropped_df if it's already collecting others
    if dropped_df is not None:
        dropped_df = pd.concat([dropped_df, dropped], axis=1)
    else:
        dropped_df = dropped

    # Make sure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    dropped_df.to_csv(save_path, index=False)

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