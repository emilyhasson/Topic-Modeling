# import pandas as pd
# from gensim.models import AuthorTopicModel
# from gensim.corpora import Dictionary
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk
# from gensim.models import TfidfModel
# from nltk.stem import WordNetLemmatizer


# # Ensure you have the necessary NLTK datasets
# nltk.download('punkt')
# nltk.download('stopwords')


# # Step 1: Read the Excel file
# df = pd.read_excel('cleaned-categories.xlsx')

# # Step 2: Preprocess the data
# stop_words = set(stopwords.words('english'))

# def preprocess(text):
#     # Check if the text is a string
#     if not isinstance(text, str):
#         return []
#     tokens = word_tokenize(text.lower())
#     return [word for word in tokens if word.isalnum() and word not in stop_words]

# df['tokens'] = df['Text'].apply(preprocess)

# # Step 3: Prepare the dictionary and corpus
# dictionary = Dictionary(df['tokens'])
# corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# # Step 4: Map industries to documents
# author2doc = {}
# for i, row in df.iterrows():
#     industry = row['Industry']
#     if industry not in author2doc:
#         author2doc[industry] = []
#     author2doc[industry].append(i)

# # Step 5: Train the Author-Topic Model
# model = AuthorTopicModel(corpus=corpus, author2doc=author2doc, id2word=dictionary, num_topics=5)

# # Step 6: Examine the results
# # Print the topics from the model
# topics = model.print_topics(num_words=5)
# for topic in topics:
#     print(topic)



# import matplotlib.pyplot as plt

# # Define a function to plot top terms for each topic
# def plot_top_terms_per_topic(model, num_terms=5):
#     for topic_num in range(model.num_topics):
#         plt.figure(figsize=(8, 3))
#         topic_terms = model.get_topic_terms(topicid=topic_num, topn=num_terms)
#         terms = [model.id2word[term_id] for term_id, _ in topic_terms]
#         weights = [weight for _, weight in topic_terms]
#         plt.barh(terms, weights, color='skyblue')
#         plt.xlabel("Weight")
#         plt.title(f"Top {num_terms} Terms in Topic {topic_num + 1}")
#         plt.gca().invert_yaxis()
#         plt.show()

# # Call the function to plot the top terms
# plot_top_terms_per_topic(model)



# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # Assuming 'model' is your trained AuthorTopicModel
# # First, we need to get the topic distributions from the model
# topic_distributions = model.state.get_lambda()

# # Normalize these distributions
# topic_distributions /= topic_distributions.sum(axis=0)

# # Map each industry to its corresponding topic distribution
# industry_topic_distributions = {industry: topic_distributions[:, model.author2id[industry]] for industry in model.id2author.values()}

# # Create a DataFrame for easier plotting
# df = pd.DataFrame(industry_topic_distributions).T  # Transpose to have industries as rows
# df.columns = [f'Topic {i+1}' for i in range(topic_distributions.shape[0])]  # Rename columns as topics

# df.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
# plt.title('Topic Distribution Across Industries')
# plt.ylabel('Proportion of Topics')
# plt.xlabel('Industries')
# plt.xticks(rotation=45)
# plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()



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


# # Function to preprocess text
# def preprocess(text):
#     result = []
#     for token in gensim.utils.simple_preprocess(text):
#         if token not in STOPWORDS and len(token) > 3:
#             result.append(lemmatizer.lemmatize(token))
#     return result

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

# import matplotlib.pyplot as plt

# # Define a function to plot top terms for each topic
# def plot_top_terms_per_topic(model, num_terms=5):
#     for topic_num in range(model.num_topics):
#         plt.figure(figsize=(8, 3))
#         topic_terms = model.get_topic_terms(topicid=topic_num, topn=num_terms)
#         terms = [model.id2word[term_id] for term_id, _ in topic_terms]
#         weights = [weight for _, weight in topic_terms]
#         plt.barh(terms, weights, color='skyblue')
#         plt.xlabel("Weight")
#         plt.title(f"Top {num_terms} Terms in Topic {topic_num + 1}")
#         plt.gca().invert_yaxis()
#         plt.show()

# # Call the function to plot the top terms
# plot_top_terms_per_topic(model)



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
