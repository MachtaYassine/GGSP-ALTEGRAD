import os
import re
import pandas as pd
import sys

# Path to input and output files
GROUPED_TRAIN_PATH = 'data/data/train/grouped_description.txt'
TRAIN_CSV_PATH = 'data/data/train/train_graph_parameters.csv'
TEST_CSV_PATH = 'data/data/test/test_graph_parameters.csv'
TXT_TRAIN_PATH = 'data/data/train/description'
TXT_TEST_PATH = 'data/data/test/test.txt'

# Regex patterns for extracting graph parameters
PATTERNS = {
    'n_nodes': r'(\d+) nodes',
    'n_edges': r'(\d+) edges',
    'avg_degree': r'(?:average degree(?: is equal to)?|each node is connected to) ([\d.]+)',
    'n_triangles': r'(\d+) triangles',
    'global_clustering_coefficient': r'(?:global clustering coefficient(?: is| and the graph\'s)?|The global clustering coefficient and the graph\'s maximum k-core are)\s+([\d.]+)(?:\s+and [\d.]+ respectively)?',
    'max_k_core': r'(?:maximum k-core(?: is| and)?|graph has a maximum k-core of|The global clustering coefficient and the graph\'s maximum k-core are [\d.]+ and)\s+([\d.]+)(?:\s+respectively)?',
    'n_communities': r'number of communities(?: equal to)? (\d+)|graph consists of (\d+) communities'
}

def read_file(file_path):
    """Reads the file and returns its lines."""
    with open(file_path, 'r') as file:
        return file.readlines()

def write_to_file(file_path, data, mode='w'):
    """Writes data to a file."""
    with open(file_path, mode) as file:
        file.write(data)

def create_grouped_file(txt_path, grouped_txt_path):
    """Crawls through the txt files and groups the descriptions into one file."""
    if os.path.exists(grouped_txt_path):
        os.remove(grouped_txt_path)
        print('Old grouped file removed')

    txt_files = [os.path.join(txt_path, f) for f in os.listdir(txt_path) if f.endswith('.txt')]
    
    for file in txt_files:
        lines = read_file(file)
        write_to_file(grouped_txt_path, lines[0], mode='a')
    print('New grouped file created')

def extract_parameters_from_sentence(sentence):
    """Extracts graph parameters from a single sentence using regex."""
    result = {}
    for param, pattern in PATTERNS.items():
        match = re.search(pattern, sentence)
        if match:
            # Use the first matched group that's not None and strip any trailing periods
            value = match.group(1) or match.group(2)
            value = value.rstrip('.')  # Remove any trailing period
            result[param] = float(value) if '.' in value else int(value)
        else:
            result[param] = None  # Default to None if not found
    return result

def extract_features_from_sentences(sentences):
    """Extracts features from a list of sentences."""
    features = []
    for sentence in sentences:
        result = extract_parameters_from_sentence(sentence)
        if None in result.values():
            raise ValueError(f"Failed to extract all parameters from sentence: {sentence}")
        features.append(result)
    return features

def convert_to_dataframe(features):
    """Converts extracted features into a pandas DataFrame."""
    return pd.DataFrame(features)

def process_train_data():
    """Process and save the train data."""
    sentences = read_file(GROUPED_TRAIN_PATH)
    features = extract_features_from_sentences(sentences)
    df = convert_to_dataframe(features)
    df.to_csv(TRAIN_CSV_PATH, index=False)
    print(f"Train data saved to {TRAIN_CSV_PATH}")

def process_test_data():
    """Process and save the test data."""
    sentences = read_file(TXT_TEST_PATH)
    features = extract_features_from_sentences(sentences)
    df = convert_to_dataframe(features)
    df.to_csv(TEST_CSV_PATH, index=False)
    print(f"Test data saved to {TEST_CSV_PATH}")

def main():
    """Main function that handles the full pipeline."""
    try:
        create_grouped_file(TXT_TRAIN_PATH, GROUPED_TRAIN_PATH)
        process_train_data()
        process_test_data()
    except ValueError as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
