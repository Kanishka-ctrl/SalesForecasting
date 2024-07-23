# load_data.py

import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

if __name__ == "__main__":
    data = load_data('data/data.csv')
    print(data.columns)  # Display column names
    print(data.head())
