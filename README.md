# AI-Semantic-Search-Engine
The aim of this project is to implement a semantic search using artificial intelligence.
To develop a search engine that encodes the user's query into a vector and searches for similarity within a body of text. The user can store all the text to be searched using a vector database like Pinecone. This search engine will be designed to provide accurate and relevant search results.


This is a powerful and common combination for building semantic search, question-answering, threat-detection, and other applications that rely on NLP and search over a large corpus of text data.
The basic workflow looks like this:

# Embed and index
Use the OpenAI Embedding API to generate vector embeddings of your documents (or any text data).
Upload those vector embeddings into Pinecone, which can store and index millions/billions of these vector embeddings, and search through them at ultra-low latencies.

# Search
Pass your query text or document through the OpenAI Embedding API again.
Take the resulting vector embedding and send it as a query to Pinecone.
Get back semantically similar documents, even if they don't share any keywords with the query.

![Alt Text](6a3ea5a-pinecone-openai-overview.png)

# Features
- Semantic search engine that provides accurate and relevant search results
- Uses artificial intelligence to encode the user's query into a vector and search for similarity within a body of text
- Can store all the text to be searched using a vector database like Pinecone
- Designed to be fast, efficient, and scalable
- Can be easily integrated into existing applications and websites

# Technologies
* Python
* Hugging Face Transformers Dataset
* Pinecone Vector Database
* OpenAI embeddings

# Usage
To use this semantic search engine, follow these steps:

1. Install the required dependencies by running pip install -r requirements.txt
2. Store the text to be searched in a vector database like Pinecone
3. Run the search engine by running python semantic_search.py or the google colab notebook can also be directly acessed
4. Enter your search query and view the search results
