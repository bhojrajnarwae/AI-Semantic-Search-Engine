import openai
import os
import pinecone
from datasets import load_dataset 
from tqdm.auto import tqdm 

openai.api_key = "<<enter API key>>"                  # enter Api key
# get API key from top-right dropdown on OpenAI website

openai.Engine.list()            # check we have authenticated or not
openai.api_key = "<<enter API key>>"           # Personal API

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

MODEL = "text-embedding-ada-002"                            # text embedded model

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=MODEL
)


# extract embeddings to a list
embeds = [record['embedding'] for record in res['data']]
print(embeds)


# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key="<<Enter Pinecone API>>",                                      # Enter Your Pinecone API
    environment= "<<Enter Environment>>"                         # find next to API key in console <<Enter Environment>>
)

# check if 'openai' index already exists (only create index if not)
if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=len(embeds[0]))      # creating index named openai in pinecone vector database
# connect to index
index = pinecone.Index('openai')


# Loading trec dataset from Hugging Face dataset 
# load the first 2K rows of the TREC dataset
trec = load_dataset('trec', split='train[:2000]')
trec

# this is our progress bar
batch_size = 35                                             # process everything in batches of 35
for i in tqdm(range(0, len(trec['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(trec['text']))
    # get batch of lines and IDs
    lines_batch = trec['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))


query = "What caused the 1929 Great Depression?"                       # query searched in dataset
xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")


query = "What were the popular songs in the early 20th century?"       # query searched in dataset
# create the query embedding
xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

# query, returning the top 10 most similar results
res = index.query([xq], top_k=10, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")


"""Let's perform one final search using the definition of songs rather than the word or related words."""
query = "What were the popular act or art of singing in the early 20th century?"

# create the query embedding
xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

# query, returning the top 10 most similar results
res = index.query([xq], top_k=10, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")

"""**Working On Query**
---
Now we can input our query to check the working of search Engine.
"""

query = input("Enter a query: ")                                          # Input the query to be searched

xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")

"""Matching the indices of query in metadata."""

res                                                           # matching the query with indices in pinecone with gpt-3

"""# **Generating Prompts**
---
We will be generating prompts to check if the serach engine could provide the releated context of query.

"""

import openai                                                            # Generating Prompt
openai.api_key="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"
def create_prompt(query):
    header = "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text and requires some latest information to be updated, print 'Sorry Not Sufficient context to answer query' \n"
    return header + query + "\n"

def generate_answer(prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop = [' END']
    )
    return (response.choices[0].text).strip()

prompt = create_prompt(query)                                  # Displaying generated prompt 
print(prompt)

"""Final reply from Engine on provided prompt."""

reply = generate_answer(prompt)                 # Printing the reply of generated prompt using gpt-3 model
print(reply)