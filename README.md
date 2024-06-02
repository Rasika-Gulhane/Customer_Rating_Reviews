# Customer Rating and Reviews 

- Sentiment Analysis with NLP 
- AI RAG Chatbot for review related queries on huge dataset


The data is reviews for a generic online retailer. We would like you to focus on the following items:

- Generate detailed insights that go beyond overall sentiment
- Identify pain points and what are working/not working for the products
- Write readable and reusable code
- added Streamlit UI
- Added Generative AI RAG pipeline for Chatbot that give complete information of data and Sentiment analysis


How to run the project for UI:

**Data Analysis and visualization with NLP and Visalization technique:**
1. Inside the python/ conda environmnet install all required libraries
`
pip install -r requirements.txt

`
`
run strealit source/data_preprocessing.py
`

**Ratinga nd Sentiment Reviews Chatbot**

1. Create Vecore database using AstraDB for data storage and similarity meassures
- Login to AstraDB and create database collectionand your accessing keys as follows:
    OPENAI_API_KEY= 'xyz'
    ASTRA_DB_API_ENDPOINT= 'xyz'
    ASTRA_DB_APPLICATION_TOKEN= 'xyz'
    ASTRA_DB_KEYSPACE= default_keyspace

2. Add this keys to environment by creating file .env
3. To chat with AI chatbot:
`
run python app.py
`

