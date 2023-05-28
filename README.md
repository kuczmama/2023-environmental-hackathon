# Answrly Hackathon 2023

## Inspiration

Everyone has different bespoke problems...

They want to:

- Summarize Scientific Papers
- Explain Policy documents
- Get main points of Phone calls
- Explain any new documents

## What it does

A tool that can summarize and explain any arbitrary length document.

## How we built it

We chunked a document into manageable context chunks.   Then we used chromaDB to find the relevant chunk, and then use OpenAI's APIs to create embeddings for the document, and then used the chat-gpt-3.5 turbo model to answer questions about the document.

We built  tool where we could upload and explain any arbitrary document

## Challenges we ran into

It was tricky to be able to parse a pdf, and convert it to text, and then handle any arbitrary size.

## Accomplishments that we're proud of

We can handle any arbitrary length documents, and have made a generic solution that can be used for more than the climate.

## What we learned

We learned how OpenAI's APIs work, Chroma DB, and also langchain.

## What's next for Answrly

Be able to sell answers to questions as as service.

## Demo

https://kuczmama-2023-environmental-hackathon-main-soul2n.streamlit.app/

## How To run

Use OpenAI to answer questions about PDF documents: https://harishgarg.com/writing/build-a-chatgpt-for-your-pdf-documents/

To run

```txt
pipenv shell
pipenv install
python main.py
```

### To run with streamlit

`streamlit run main.py`


References: https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9

The frontend uses streamlit 
