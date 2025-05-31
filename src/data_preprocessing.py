import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
nltk.download('stopwords') # Downloading stopwords data
nltk.download('punkt') # Downloading tokenizer data
nltk.download('wordnet')

# Ensure "logs" exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Setting up the logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    # Transform the input text by converting it to lowercase, tokenizing, removing stopwords and punctuations and stemming.
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non alphanumeric tokens
    text =  [ word for word in text if word.isalnum() ]
    # Remove stopwords and punctuations
    text = [ word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preprocess_df(df, text_column = 'text', target_column = 'target'):
    # Preprocess the dataframe by encoding the target column, removing duplicates and transforming the text column.
    try:
        logger.debug('Starting Preprocessing of DataFrame.')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded.')

        # Remove duplicate rows
        df = df.drop_duplicates(keep= 'first')
        logger.debug("Duplcate rows have been removed.")

        # Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed.')
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column = 'text', target_column = 'target'):
    # Main function to load row data, process it and then save the processed data.
    try:
        #Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Data loaded properly.")

        #Transform the dat
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        #Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug("Processed data saved to %s", data_path)

    except FileNotFoundError as e:
        logger.error("file not found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
    except Exception as e:
        logger.error("Failed to complete the data tranformation process: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()