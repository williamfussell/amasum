import pandas as pd
import numpy as np
import gzip
import json

def parse(path):
    """
    Parses a Gzip-compressed JSON file and yields JSON objects one by one.

    Parameters:
    path (str): The path to the Gzip-compressed JSON file.

    Yields:
    dict: A JSON object read from the file.
    """
    # Open the Gzip-compressed file in binary read mode
    g = gzip.open(path, 'rb')

    # Iterate through each line (JSON object) in the file
    for l in g:
        # Parse and load the JSON object from the current line
        json_object = json.loads(l)

        # Yield the JSON object to the caller one by one
        yield json_object

def getDF(path):
    """
    Reads a Gzip-compressed JSON file and returns a pandas DataFrame.

    Parameters:
    path (str): The path to the Gzip-compressed JSON file.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the JSON file.
    """
    # Initialize a counter for DataFrame index
    i = 0

    # Create an empty dictionary to store JSON objects
    df = {}

    # Iterate through each JSON object returned by the 'parse' function
    for d in parse(path):
        # Add the JSON object to the dictionary with the current index as the key
        df[i] = d

        # Increment the index counter for the next JSON object
        i += 1

    # Create a pandas DataFrame from the dictionary
    # Using 'orient='index'' arranges the DataFrame where each row corresponds to an index (i)
    return pd.DataFrame.from_dict(df, orient='index')

# Example usage of the 'getDF' function:
# 'path' is the path to the Gzip-compressed JSON file
# df = getDF('path')