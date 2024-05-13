import pandas as pd

# Load the dataset
file_path = 'cleaned-categories.xlsx'
data = pd.read_excel(file_path)


# Extract the column with the text
texts = data['Text'].dropna()


import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatizer.lemmatize(token))
    return result

# Apply preprocessing
processed_texts = texts.apply(preprocess)

from gensim import corpora, models

# Create a dictionary representation of the documents.
dictionary = corpora.Dictionary(processed_texts)

# Filter out extremes to remove noise
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Convert document into the bag-of-words (BoW) format
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# Set up the LDA model
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

# View the topics in LDA model
lda_topics = lda_model.print_topics(num_words=5)
for topic in lda_topics:
    print(topic)



import pyLDAvis
import pyLDAvis.gensim


# # Prepare the visualization
lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)


# Or directly show in the browser
pyLDAvis.display(lda_display)

pyLDAvis.save_html(lda_display, 'lda_visualization.html')
