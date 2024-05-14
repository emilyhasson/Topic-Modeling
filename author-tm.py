

import pandas as pd
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import nltk
from gensim import corpora, models

nltk.download('wordnet')

# Load the dataset
file_path = 'cleaned-categories.xlsx'
data = pd.read_excel(file_path).dropna(subset=['Text', 'Industry'])  # Ensure no NA values in key columns

# Extract the column with the text and industry
texts = data['Text']
industries = data['Industry']

lemmatizer = WordNetLemmatizer()


# Define custom stopwords or ignore list
# custom_stopwords = {'think', 'year', 'data', 'technology', 'customer'}
custom_stopwords = {}
all_stopwords = STOPWORDS.union(custom_stopwords)

# Function to preprocess text
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in all_stopwords and len(token) > 3:
            result.append(lemmatizer.lemmatize(token))
    return result



# Apply preprocessing
processed_texts = texts.apply(preprocess)

# Create a dictionary representation of the documents.
dictionary = corpora.Dictionary(processed_texts)

# Filter out extremes to remove noise
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Convert document into the bag-of-words (BoW) format
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# Map industries to documents
author2doc = {}
for i, industry in enumerate(industries):
    if industry not in author2doc:
        author2doc[industry] = []
    author2doc[industry].append(i)

# Set up the Author-Topic Model
author_model = models.AuthorTopicModel(corpus=corpus, author2doc=author2doc, id2word=dictionary, num_topics=10, random_state=100, iterations=50, alpha='auto')

# View the topics in the Author-Topic Model
author_topics = author_model.print_topics(num_words=5)
for topic in author_topics:
    print(topic)

model = author_model


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'model' is your trained AuthorTopicModel
# First, we need to get the topic distributions from the model
topic_distributions = model.state.get_lambda()

# Normalize these distributions
topic_distributions /= topic_distributions.sum(axis=0)

# Map each industry to its corresponding topic distribution
industry_topic_distributions = {industry: topic_distributions[:, model.author2id[industry]] for industry in model.id2author.values()}

# Create a DataFrame for easier plotting
df = pd.DataFrame(industry_topic_distributions).T  # Transpose to have industries as rows
df.columns = [f'Topic {i+1}' for i in range(topic_distributions.shape[0])]  # Rename columns as topics

df.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.title('Topic Distribution Across Industries')
plt.ylabel('Proportion of Topics')
plt.xlabel('Industries')
plt.xticks(rotation=45)
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
