# Colab Book: https://colab.research.google.com/drive/1m0Mm83_AXIIkxra18daYHEC_R4MIaQi_#scrollTo=0tupSrBk6UZW

import pandas as pd
import streamlit as st

# Load data
file_path = 'data/Raw_Reviews.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)
df = df.dropna(subset=['Text_Review','Type'])
df.isnull().sum()
df['Title'] = df['Title'].fillna('unknown')

# Display data
st.title('Product Review Analysis')
st.subheader('Data Preview')
st.write(df.head())


################################################################################################

# Text processing


import re
import string
import nltk
from nltk.corpus import stopwords

stopset = stopwords.words('english')

def clean_text(text):

    text = re.sub(f"[{string.punctuation}]", "", text)    # Remove punctuation
    text = text.lower()

    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text

# Apply the function to the 'review_text' column
df['cleaned_review'] = df['Text_Review'].apply(clean_text)

# Display data

st.subheader('1. Clean Data Preview')
st.write(df[['Text_Review', 'cleaned_review']].head())

################################################################################################


# Sentiment Analysis


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores
def get_sentiment(text):
    return sid.polarity_scores(text)

# Apply the function to the 'cleaned_review' column
df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

# Extract compound sentiment scores
df['sentiment_score'] = df['sentiment'].apply(lambda x: x['compound'])

# Display data

st.subheader('2. Sentiment Score')
st.write(df[['Text_Review', 'sentiment', 'sentiment_score']].head())


################################################################################################

# Identify Common theme in Reviews:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Create a CountVectorizer to count the word frequencies
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df['cleaned_review'])

# Fit the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)


# Display the top words in each topic
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        topics.append(f"Topic {topic_idx}: {top_words}")
    return topics

# Display data with Streamlit
st.subheader('3. Top 10 Common Themes in Reviews')

# Get the topics
topics = display_topics(lda, vectorizer.get_feature_names_out(), 10)

# Write each topic in a separate line
for topic in topics:
    st.write(topic)


################################################################################################


# Pain Point and Highlights:


# Define thresholds for positive and negative sentiment
positive_threshold = 0.5
negative_threshold = -0.5

# Extract positive and negative reviews
positive_reviews = df[df['sentiment_score'] > positive_threshold]
negative_reviews = df[df['sentiment_score'] < negative_threshold]

# Display examples of positive and negative reviews

st.subheader('4. i Positive Reviews:')
st.write(positive_reviews['Text_Review'].head())

st.subheader('4. ii Negative Reviews:')
st.write(negative_reviews['Text_Review'].head())


# Group by product type and calculate counts of positive and negative reviews
product_sentiment_counts = df.groupby('Type').agg(positive_count=('sentiment_score', lambda x: (x > positive_threshold).sum()),
                                                           negative_count=('sentiment_score', lambda x: (x < negative_threshold).sum()))

# Display product sentiment counts
st.subheader('4. iii Product Sentiment Count:')
st.write(product_sentiment_counts)

import matplotlib.pyplot as plt


# Plot the grouped bar plot
fig, ax = plt.subplots(figsize=(10, 6))
product_sentiment_counts.plot(kind='bar', ax=ax)
ax.set_title('Counts of Positive and Negative Reviews by Product Type')
ax.set_xlabel('Product Type')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend(title='Sentiment', loc='upper right', labels=['Positive', 'Negative'])
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)


################################################################################################


# Incorrect Sentiment w.r.t Rating

# Define a function to check if the rating matches the sentiment
def check_rating_match(row):
    if (row['sentiment_score'] > 0 and row['Rating'] >= 4) or (row['sentiment_score'] < 0 and row['Rating'] <= 2):
        return 'Correct'
    else:
        return 'Incorrect'

# Apply the function to each row
df['rating_match'] = df.apply(check_rating_match, axis=1)

# Display the result
print(df[['Text_Review', 'Rating', 'sentiment_score', 'rating_match']].head())


# Filter rows where ratings do not match sentiment
incorrect_ratings = df[df['rating_match'] == 'Incorrect']

# Group by product type and count the number of incorrect ratings for each type
incorrect_ratings_count = incorrect_ratings.groupby('Type').size()

# Display the counts of incorrect ratings for each type
st.subheader('5. Rating Vs Review: No Match Count')
st.write(incorrect_ratings_count)

# Group by product type and count the number of correct and incorrect ratings for each type
rating_counts = df.groupby(['Type', 'rating_match']).size().unstack(fill_value=0)

# Plot the bar plot
fig, ax = plt.subplots(figsize=(10, 6))
rating_counts.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Correct and Incorrect Ratings by Product Type')
ax.set_xlabel('Product Type')
ax.set_ylabel('Count')
ax.legend(title='Rating Match', loc='upper right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)


################################################################################################

# Sentiment wise Word Count:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Generate a word cloud for positive reviews
positive_text = " ".join(positive_reviews['cleaned_review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)


# Display the word cloud
st.subheader('6. i positive review words example')
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Positive Reviews Word Cloud')

# Display the plot in Streamlit
st.pyplot(fig)



# Generate a word cloud for negative reviews
negative_text = " ".join(negative_reviews['cleaned_review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)


# Display the word cloud
st.subheader('6. ii Negative review words example')
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Negative Reviews Word Cloud')

# Display the plot in Streamlit
st.pyplot(fig)


df.to_csv('data/modeified_raw_reviews.csv')
