import streamlit as st
import os
import pickle
import json

# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.document_loaders import DirectoryLoader

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma

# from langchain_community import embeddings

from langchain_community.embeddings import OllamaEmbeddings
# from nomic import embed

from langchain_community.llms import Ollama

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.text_splitter import CharacterTextSplitter

from langchain_core.messages import AIMessage, HumanMessage


# from langchain_community.chains import RetrievalQA


# Process PDFs


# URL processing
def process_input(question):
    model_local = Ollama(model="mistral")

    # Convert string of URLs to list
    # urls_list = urls.split("\n")
    # docs = [PyPDFLoader(pdf) for pdf in pdf_docs]
    # docs_list = [item for sublist in docs for item in sublist]

    # split the text into chunks

    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    # doc_splits = text_splitter.split_documents(docs_list)

    # convert text chunks into embeddings and store in vector database

    # vectorstore = Chroma.from_documents(
    #   documents=doc_splits,
    #  collection_name="rag-chroma",
    # embedding=OllamaEmbeddings(model='nomic-embed-text'),
    # )
    # retriever = vectorstore.as_retriever()

    # perform the RAG
    vectorstore = Chroma(persist_directory="./db", embedding_function=OllamaEmbeddings(model='nomic-embed-text'))
    # vectorstore.get(include=['embeddings', 'documents', 'metadatas'])
    retriever = vectorstore.as_retriever()

    after_rag_template = """ Answer the question based only on the following context,:{context} Question:{question} \nWhat are the top 10 most likely diagnoses? Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10).Ensure the order starts with the most likely. The top 10 diagnoses are."""
    no_rag_template = """Question:{question} \nWhat are the top 10 most likely diagnoses? Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10).Ensure the order starts with the most likely. The top 10 diagnoses are."""
   
    # print(after_rag_template)

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    no_rag_prompt = ChatPromptTemplate.from_template(no_rag_template)
    # print(after_rag_prompt)
    after_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()
    )



    no_rag_chain = (
            {"question": RunnablePassthrough()}
            | no_rag_prompt
            | model_local
            | StrOutputParser()
    )

    return after_rag_chain.invoke(question), no_rag_chain.invoke(question)

    # streamlit UI


st.title("Differential Diagnosis with Medical Text RAG")

# st.write("enter urls (one per line) and a question to query the documents")

with st.sidebar:
    st.write("Upload your Medical Text in PDF format")
    # UI for input fields

    # urls = st.text_area("Enter URLs separated by a New Line", height =150)
    UploadedFiles = st.file_uploader("Upload your Medical Textbook in PDF format here and click on 'Upload'", accept_multiple_files=True)

    # persisting the Chromadb Database

    persist_directory = "./db"

    if st.button("Upload"):
        try:
            os.mkdir("UploadedTextbook")
        except:
            print("File already exists")
        with st.spinner("Processing SOPs"):
            # get the pdf text
            DocumentList = []
            for UploadedFile in UploadedFiles:
                with open(os.path.join("UploadedTextbook", UploadedFile.name), "wb") as f:
                    f.write(UploadedFile.getbuffer())
                DocumentList.append(os.path.join("UploadedTextbook", UploadedFile.name))

            docs = [PyPDFLoader(pdf).load() for pdf in DocumentList]
            docs_list = [item for sublist in docs for item in sublist]

            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
            doc_splits = text_splitter.split_documents(docs_list)

            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                # collection_name="rag-chroma",
                embedding=OllamaEmbeddings(model='nomic-embed-text'),
                persist_directory=persist_directory,
            )
            vectorstore.persist()
            vectorstore = None
            # retriever = vectorstore.as_retriever()

        st.write("Textbook Uploaded")

question = st.text_input("Enter the Case")
# topics = st.selectbox("Enter related Chapter", ())
# button = st.text_input("Question")

if st.button('List differential diagnosis'):
    with st.spinner('Processing ....'):
        # print(DocumentList)
        DocumentList = os.listdir(r'UploadedTextbook')
        # docs = [PyPDFLoader(os.path.join("UploadedSOPs",pdf)).load() for pdf in DocumentList]
        # docs_list = [item for sublist in docs for item in sublist]

        rag_answer, no_rag_answer = process_input(question)
        # print(answer)
        st.text_area("Answer with RAG", value=rag_answer, height=300)

        st.text_area("Answer without RAG", value=no_rag_answer, height=300)

    # These comments should b removed before production
    # NER
