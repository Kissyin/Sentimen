import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import argparse

# Function to ensure NLTK data path is properly set
def set_nltk_data_path():
    import nltk.data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Ensure NLTK data path is properly set
set_nltk_data_path()

# Load positive and negative word lists
with open('positive.txt', 'r') as file:
    positive_words = file.read().splitlines()

with open('negative.txt', 'r') as file:
    negative_words = file.read().splitlines()

# Normalize the text data
norm = {
    " dgn ": " dengan ", " gue ": " saya ", " dgn ":" dengan ", "bgmn ":" bagaimana ", 
    'dk':' tidak ', 'lum ':' belum ', 'antaaaaaaaappp':' bagus ', 'osss ':' bagus ', 
    'hanks ': 'erima kasih ', 'fast':' cepat ', 'g ':' dengan ', 'trims':' terima kasih ', 
    'brg':' barang ', 'gx':' tidak ', 'gn ':' dengan ', 'ecommended':' rekomen ', 
    'ecomend':' rekomen ', 'good':' bagus '
}

def normalisasi(str_text):
    if isinstance(str_text, str):  
        for i in norm:
            str_text = str_text.replace(i, norm[i])
    return str_text

def process_file(file_path, output_dir):
    # Load the data
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path)

    # Normalize the text data
    df['Tweet'] = df['Tweet'].apply(lambda x: normalisasi(x))

    # Replace NaN values in the 'Tweet' column with an empty string
    df['Tweet'] = df['Tweet'].fillna('')

    # Tokenize the text data
    print("Tokenizing tweets...")
    tokenized_tweets = df['Tweet'].apply(word_tokenize)

    # Remove stopwords
    print("Removing stopwords...")
    stop_words = set(stopwords.words('indonesian'))
    tokenized_tweets = tokenized_tweets.apply(lambda x: [word for word in x if word not in stop_words])

    # Lemmatize the tokens
    print("Lemmatizing tokens...")
    lemmatizer = WordNetLemmatizer()
    tokenized_tweets = tokenized_tweets.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Create a function to assign sentiment scores weighted by word count
    def get_weighted_sentiment_score(tokens):
        pos_count = sum(1 for token in tokens if token in positive_words)
        neg_count = sum(1 for token in tokens if token in negative_words)
        total_count = pos_count + neg_count
        if total_count == 0:
            return 0  # Neutral if no positive or negative words
        sentiment_score = (pos_count - neg_count) / total_count
        weighted_sentiment_score = sentiment_score * len(tokens)  # Weight by word count
        return weighted_sentiment_score

    # Apply the weighted sentiment scoring function to each tweet
    print("Calculating sentiment scores...")
    df['Weighted_Sentiment_Score'] = tokenized_tweets.apply(get_weighted_sentiment_score)

    # Get the filename without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Save the output to an Excel file in the output directory
    output_file_path = os.path.join(output_dir, f"{file_name}_sentimen.xlsx")
    print(f"Saving results to: {output_file_path}")
    df.to_excel(output_file_path, index=False)
    print("Processing complete.")

# Setup argument parser
parser = argparse.ArgumentParser(description="Apply sentiment analysis to CSV files in the input directory.")
parser.add_argument('--inputdir', type=str, required=True, help='Input directory path')
parser.add_argument('--outputdir', type=str, required=True, help='Output directory path')
args = parser.parse_args()

# Process each file in the input directory
input_dir = args.inputdir
output_dir = args.outputdir

# Check if the output directory exists, and if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        input_file_path = os.path.join(input_dir, file_name)
        process_file(input_file_path, output_dir)
