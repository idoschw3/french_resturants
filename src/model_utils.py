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
      - If an object column's unique values (ignoring nulls) are only {"Y", "N"}, it is converted to boolean.
      - If the column name is in force_category, convert it to a category.
      - If the column name is in force_string, leave it as a string.
      - Else if the column has a high unique ratio (> uniqueness_threshold)
        or at least one value longer than length_threshold characters,
        leave it as a string (object dtype).
      - Otherwise, convert the column to categorical.

    Parameters:
      data: The input DataFrame.
      uniqueness_threshold: Proportion threshold for uniqueness (default 0.5).
      length_threshold: Maximum length threshold (default 30 characters).
      force_category: List of column names that should always be converted to category.
      force_string: List of column names that should always be left as string.

    Returns:
      The DataFrame with updated column types.
    """
    if force_category is None:
        force_category = []  # No forced category columns by default
    if force_string is None:
        force_string = []  # No forced string columns by default

    # Iterate over all columns in the DataFrame
    for col in data.columns:
        # Only process columns of object dtype
        if data[col].dtype == 'object':
            # Get the set of unique non-null values
            unique_vals = set(data[col].dropna().unique())

            # Rule 1: If the column's unique values are only "Y" and "N", convert to boolean.
            if unique_vals.issubset({"Y", "N"}):
                data[col] = data[col].map({"Y": True, "N": False}).astype("boolean")
            # Rule 2: If the column is forced to be a category, convert to category.
            elif col in force_category:
                data[col] = data[col].astype("category")
            # Rule 3: If the column is forced to be left as string, convert to string.
            elif col in force_string:
                data[col] = data[col].astype(str)
            else:
                # Calculate the unique ratio and maximum string length.
                unique_ratio = data[col].nunique() / len(data[col])
                max_length = data[col].str.len().max()
                # If high unique ratio or long values, leave as string; otherwise, convert to category.
                if unique_ratio > uniqueness_threshold or max_length > length_threshold:
                    data[col] = data[col].astype(str)
                else:
                    data[col] = data[col].astype("category")
    return data