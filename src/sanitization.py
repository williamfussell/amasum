import pandas as pd
import numpy as np
import gzip
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')  # Download the necessary resource for tokenization
nltk.download('stopwords') 
nltk.download('wordnet')  


################################################################################################################
#READ IN DATA

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


################################################################################################################
#CLEANING

def clean_amazon_reviews(df):
    """
    Clean the review text in a pandas DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the review data.

    Returns:
        pandas.DataFrame: The DataFrame with cleaned review text added as 'cleaned_review' column.
    """

    # Define text cleaning functions
    def remove_html_tags(text):
        """Remove HTML tags from the text."""
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text
    
    def remove_special_characters(text):
        """Remove non-alphanumeric characters from the text."""
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
        return clean_text
    
    def convert_to_lowercase(text):
        """Convert text to lowercase."""
        clean_text = text.lower()
        return clean_text
    
    # Apply text cleaning functions
    df['cleaned_review'] = df['reviewText'].apply(remove_html_tags)
    df['cleaned_review'] = df['cleaned_review'].apply(remove_special_characters)
    df['cleaned_review'] = df['cleaned_review'].apply(convert_to_lowercase)
    
    return df



################################################################################################################
#TOKENIZATION


def tokenize_text(df, text_column):
    """
    Tokenize the text in a DataFrame column into individual words or tokens.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text to be tokenized.
        text_column (str): The name of the column containing the text to be tokenized.

    Returns:
        pandas.DataFrame: The original DataFrame with an additional column 'tokens' containing the tokenized text.
    """
    # Ensure the DataFrame has the specified column
    if text_column not in df:
        raise ValueError(f"'{text_column}' column not found in the DataFrame.")
    
    # Tokenize the text using NLTK's word_tokenize and create a new column 'tokens'
    df['tokens'] = df[text_column].apply(lambda text: word_tokenize(text))
    return df

# # Example usage
# # Assuming 'cleaned_df' is your DataFrame with a 'cleaned_review' column
# tokenized_df = tokenize_text(cleaned_df, 'cleaned_review')
# print(tokenized_df.head())

################################################################################################################
#STOP WORD REMOVAL

def remove_stop_words(df, text_column):
    """
    Remove common stop words from text in a DataFrame column.
    
    This function tokenizes the text in the specified column and removes common stop words
    from the tokens using NLTK's English stop words corpus.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the text to process.
        text_column (str): The name of the column containing the text to process.
        
    Returns:
        pandas.DataFrame: The original DataFrame with an additional column containing tokens with stop words removed.
    """
    # Load the set of English stop words
    stop_words = set(stopwords.words('english'))
    
    # Tokenize the text in the specified column and remove stop words
    df['tokens_without_stop_words'] = df[text_column].apply(lambda text: [token for token in word_tokenize(text) if token.lower() not in stop_words])
    
    return df

# Example usage
# Assuming 'tokenized_df' is your DataFrame with a 'cleaned_review' column
# tokenized_df = remove_stop_words(tokenized_df, 'cleaned_review')
# print(tokenized_df.head())


################################################################################################################
#LEMMATIZATION

