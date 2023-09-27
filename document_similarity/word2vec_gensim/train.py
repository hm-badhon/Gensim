from gensim.models import Doc2Vec
from data_processing import preprocessed_documents

print('Train a Doc2Vec model')
model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=3)

print(f"Total training examples: {model.corpus_count}")
print(f"Total epochs: {model.epochs}")

model.build_vocab(preprocessed_documents)
model.train(preprocessed_documents, total_examples=model.corpus_count, epochs=model.epochs)

# Save the Doc2Vec model
model_save_path = "models/doc2vec_model_test.model"
model.save(model_save_path)
print('Successfully model saved')