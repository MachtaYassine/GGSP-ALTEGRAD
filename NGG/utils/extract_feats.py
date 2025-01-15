
import random
import re

random.seed(32)


def extract_numbers(text):

    """
    Extract numbers (integers and floats) from a given text using regex.

    Args:
        text (str): Input text to extract numbers from.

    Returns:
        list: A list of extracted numbers as floats.
    """
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    return [float(num) for num in numbers]

def extract_feats(file, exclude_features=None):
    """
    Extract features from a text file while excluding specified features.

    Args:
        file (str): Path to the text file.
        exclude_features (list): List of substrings indicating features to exclude.

    Returns:
        list: A list of extracted feature values as floats.
    """
    if exclude_features is None:
        exclude_features = []

    stats = []
    with open(file, "r") as fread:
        lines = fread.readlines()
        for line in lines:
            # Skip lines that contain any excluded feature
            if any(feature in line for feature in exclude_features):
                continue
            # Extract numbers from the remaining lines
            stats.extend(extract_numbers(line))
    return stats