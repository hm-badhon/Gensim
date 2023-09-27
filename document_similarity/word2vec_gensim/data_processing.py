import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models.doc2vec import TaggedDocument

# Function to preprocess a document
def preprocess_document(document):
    # Step 1: Text Cleaning
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', document)

    # Step 2: Tokenization
    sentences = sent_tokenize(cleaned_text)
    words = word_tokenize(cleaned_text)

    # Step 3: Stopword Removal
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    return filtered_words  # Return the preprocessed words as a list

# Define the path to the 20 Newsgroups dataset
dataset_path = 'dataset/20news-19997/20_newsgroups'

# Initialize a list to store preprocessed documents as lists of words
preprocessed_documents = []

# Iterate through the dataset folders and preprocess documents
for category in os.listdir(dataset_path):
    category_folder = os.path.join(dataset_path, category)
    if os.path.isdir(category_folder):
        for document_name in os.listdir(category_folder):
            document_path = os.path.join(category_folder, document_name)
            if os.path.isfile(document_path):
                with open(document_path, 'r', encoding='latin1') as file:
                    document_content = file.read()
                    preprocessed_words = preprocess_document(document_content)
                    preprocessed_documents.append(TaggedDocument(preprocessed_words, [document_path]))

print(" Successfully data pre-processing")

