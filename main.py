# import the modules
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pickle
import os
import os.path
import streamlit as st

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = 'gpt-3.5-turbo'

st.title("Ask questions about your document")
uploaded_file = st.file_uploader("Choose a file", "pdf")
query = st.text_input(
    "Enter your text query",
    "",
    key="placeholder",
)
embeddings = ''

if st.button("Execute Query") and query:
    pickle_file_name = "{}.pkl".format(uploaded_file.name)
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    pickle_file_exists = os.path.exists(pickle_file_name)
    if pickle_file_exists:
        with open(pickle_file_name, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print(f"Creating new enumeration")
        embeddings = OpenAIEmbeddings()
        with open(pickle_file_name, "wb") as f:
            pickle.dump(embeddings, f)

    docsearch = Chroma.from_texts(texts, embeddings)
    chain = load_qa_chain(
        OpenAI(temperature=0, api_key=OPENAI_API_KEY, model_name=MODEL_NAME), chain_type="stuff"
    )
    docs = docsearch.similarity_search(query)
    query_result = chain.run(input_documents=docs, question=query)
    st.write(query_result)
    st.write(docs)
