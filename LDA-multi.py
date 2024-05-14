import pandas as pd
import gensim
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import nltk
import pyLDAvis
import pyLDAvis.gensim

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


# Function to preprocess text
def preprocess(text):
    if not isinstance(text, str):
        text = str(text)  # Convert non-string text to string
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatizer.lemmatize(token))
    return result

file_path = 'cleaned-categories.xlsx'
data = pd.read_excel(file_path)



data1 = data[data['Industry'] == "Information Technology"]
data2 = data[data['Industry'] == "Communication Services"]

texts1 = data1['Text'].dropna()
texts2 = data2['Text'].dropna()
processed_texts1 = texts1.apply(preprocess).tolist()
processed_texts2 = texts2.apply(preprocess).tolist()

texts_combined = processed_texts1 + processed_texts2  # Append the two lists of preprocessed texts

dictionary = corpora.Dictionary(texts_combined)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)






def LDA(processed_texts, industry):

    # Convert document into the bag-of-words (BoW) format
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Set up the LDA model
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    return lda_model


lda_model1 = LDA(processed_texts1, "Information Technology")
lda_model2 = LDA(processed_texts2, "Communication Services")


# Assume lda_model1 and lda_model2 are your two LDA models from different industries
topic_vectors1 = [lda_model1.get_topic_terms(tid, topn=None) for tid in range(lda_model1.num_topics)]
topic_vectors2 = [lda_model2.get_topic_terms(tid, topn=None) for tid in range(lda_model2.num_topics)]

# Convert topic vectors to dense format for easier comparison
import numpy as np

def topics_to_dense(topic_vector, num_words):
    dense_vector = np.zeros(num_words)
    for word_id, prob in topic_vector:
        dense_vector[word_id] = prob
    return dense_vector

num_words = len(dictionary)  # Assuming the same dictionary was used for both models
dense_vectors1 = [topics_to_dense(tv, num_words) for tv in topic_vectors1]
dense_vectors2 = [topics_to_dense(tv, num_words) for tv in topic_vectors2]


from scipy.spatial.distance import cosine

# Calculate cosine similarity for each pair of topics
similarity_matrix = np.zeros((lda_model1.num_topics, lda_model2.num_topics))
for i, vec1 in enumerate(dense_vectors1):
    for j, vec2 in enumerate(dense_vectors2):
        similarity_matrix[i, j] = 1 - cosine(vec1, vec2)  # cosine function returns the distance

# Optional: print the matrix or visualize it
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm')
plt.title('Cosine Similarity Between Topics of Two Industries')
plt.xlabel('Topics from Information Technology')
plt.ylabel('Topics from Communication Services')
plt.show()
