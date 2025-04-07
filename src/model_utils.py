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