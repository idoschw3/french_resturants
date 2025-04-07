import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(data, column_to_check, columns_to_drop=None, train_size=0.7):
    """
    Prepares the dataset by splitting it into training and testing sets.

    Parameters:
    - data: The dataset (DataFrame).
    - column_to_check: The target variable (dependent variable).
    - columns_to_drop: Optional; additional columns to drop before training.
    - train_size: The proportion of data to use for training (default 70%).

    Returns:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training target values
    - y_test: Testing target values
    """

    drop = []
    if columns_to_drop:
        # Ensure columns_to_drop is a list
        drop = [columns_to_drop] if isinstance(columns_to_drop, str) else columns_to_drop

    # X: Features (drop specified columns + target column)
    X = data.drop(drop + [column_to_check], axis=1)

    # y: Target variable (dependent variable)
    y = data[column_to_check]

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, train_size=train_size, stratify=y
    )

    return X_train, X_test, y_train, y_test


def convert_object_columns(data, uniqueness_threshold=0.5, length_threshold=30, force_category=None, force_string=None):
    """
    Converts object columns in a DataFrame to the desired types without handling nulls.

    Rules:
      - If an object column's unique (non-null) values are exactly {"Y", "N"}, convert it to boolean.
      - If the column name is in force_string, leave it as a string.
      - If the column name is in force_category, convert it to a category.
      - Otherwise, calculate:
            unique_ratio = (number of unique values / total rows)
            max_length = maximum string length in the column.
        If unique_ratio > uniqueness_threshold OR max_length > length_threshold,
            leave the column as a string.
        Otherwise, convert the column to categorical.

    Parameters:
      data: The input DataFrame.
      uniqueness_threshold: Proportion threshold for uniqueness (default 0.5).
      length_threshold: Maximum length threshold (default 30 characters).
      force_category: List of column names to always convert to category.
      force_string: List of column names to always leave as string.

    Returns:
      The DataFrame with updated column types.
    """
    if force_category is None:
        force_category = []
    if force_string is None:
        force_string = []

    for col in data.columns:
        if data[col].dtype == 'object':
            # If the column is forced to be left as string, convert it to string immediately.
            if col in force_string:
                data[col] = data[col].astype(str)
            # Else, if the column is forced to be a category, do so.
            elif col in force_category:
                data[col] = data[col].astype("category")
            else:
                # If the column's non-null unique values are only "Y" and "N", convert to boolean.
                unique_vals = set(data[col].dropna().unique())
                if unique_vals.issubset({"Y", "N"}):
                    data[col] = data[col].map({"Y": True, "N": False}).astype("boolean")
                else:
                    # For the remaining columns, calculate the unique ratio and maximum string length.
                    series_str = data[col].astype(str)
                    unique_ratio = series_str.nunique() / len(series_str)
                    max_length = series_str.str.len().max()
                    # If high unique ratio or long strings, leave as string.
                    if unique_ratio > uniqueness_threshold or max_length > length_threshold:
                        data[col] = series_str  # leave as string
                    else:
                        # Otherwise, convert to category.
                        data[col] = data[col].astype("category")
    return data