from gensim.models import Doc2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved Doc2Vec model
model_save_path = "models/doc2vec_model.model"
model = Doc2Vec.load(model_save_path)

# Sample sentences to test
sample_sentence1 = "This is a test sentence."
sample_sentence2 = "This is a test sentence."
print(sample_sentence1)
print(sample_sentence2)

# Preprocess the sample sentences
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_sentence(sentence):
    words = word_tokenize(sentence.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

sample_sentence1_words = preprocess_sentence(sample_sentence1)
sample_sentence2_words = preprocess_sentence(sample_sentence2)

# Infer vectors for the sample sentences
vector1 = model.infer_vector(sample_sentence1_words)
vector2 = model.infer_vector(sample_sentence2_words)

# Calculate cosine similarity
similarity_score = cosine_similarity([vector1], [vector2])[0][0]

print(f"Similarity between the two sample sentences: {similarity_score:.2f}")
