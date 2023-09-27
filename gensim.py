from gensim import corpora
documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# First, let’s tokenize the documents, remove common words (using a toy stoplist) as well as words that only appear once in the corpus:

stoplist = set('for a of the and to in'.split())
type_stopList = type(stoplist)
print("type : ", type_stopList)
print("stop list "stoplist)

texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
print(texts)


# remove 