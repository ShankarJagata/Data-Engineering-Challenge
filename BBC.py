# Import necessary libraries
import os #pip install os
import zipfile #pip install zipfile
import pandas as pd #pip install pandas
import nltk #pip install nltk
from nltk.corpus import stopwords #pip install scikit-learn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# Extract the data.zip file
with zipfile.ZipFile("D:/Programs/test/data.zip", 'r') as zip_ref:
    zip_ref.extractall()

# Create an empty list to store article data
article_data = []

# Loop through each text file in the BBC_articles folder
for filename in os.listdir("D:/Programs/test/BBC_articles"):
    if filename.endswith(".txt"):
        # Extract article ID and category from filename
        article_id, category = filename.split('_')
        category = category[:-4]
        article_id = int(article_id)  # Convert article ID to integer
        
        # Read text content from the file
        with open(os.path.join("D:/Programs/test/BBC_articles", filename), 'r') as file:
            text = file.read()
        
        # Append article data to the list
        article_data.append({'article_id': article_id, 'text': text, 'category': category})
        #print(article_data)

# Create a DataFrame from the list and save as CSV
df = pd.DataFrame(article_data)
#print(df)
df.to_csv('bbc_articles.csv', index=False)


# Read the csv file into a DataFrame
# This is the first step where we load our data from a CSV file.
df = pd.read_csv('D:/Programs/test/bbc_articles.csv')

# Download NLTK stopwords
# Stopwords are common words that do not contribute much to the content or meaning of a document (e.g., "the", "and", "is").
nltk.download('punkt')
nltk.download('stopwords')

# Tokenize the text data
# Tokenization is the process of breaking down text into words, phrases, symbols, or other meaningful elements called tokens.
df['tokenized_text'] = df['text'].apply(word_tokenize)

# Perform preprocessing steps
# Preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it.
stop_words = set(stopwords.words('english'))
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word.lower() for word in x if word.isalpha() and word not in stop_words])

# Convert tokenized text back to string
# This is done to prepare the text data for vectorization.
df['processed_text'] = df['tokenized_text'].apply(' '.join)

# Vectorize the text data using TF-IDF
# TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic used to reflect how important a word is to a document in a collection or corpus.
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(df['processed_text'])

# Convert the matrix to a DataFrame and concatenate with the original DataFrame
# This is done to prepare the data for machine learning models.
features_df = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out())
df = pd.concat([df, features_df], axis=1)

# Write the DataFrame to a new csv file
# The preprocessed and vectorized data is saved to a new CSV file.
df.to_csv('vectorized_dataset.csv', index=False)