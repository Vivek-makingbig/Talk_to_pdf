# # Load the pdf and extracting the text from it
# from pypdf import PdfReader

# #load the doc
# document = PdfReader("pdfs/sample.pdf")

# #Iterate through the pages of the pdf and extract the text
# fulltext=""
# for page in document.pages:
#     fulltext += page.extract_text()

# # print("The text of the ducment is shown below:\n")
# # print(fulltext)



# #Now chunking the text data

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,  
#     chunk_overlap=200
# )

# chunks = text_splitter.split_text(fulltext)

# print(f"The size of chunks: {len(chunks)}")
# # print("\n The first chunk is:",chunks[0])


# #Creating embeddings for the text data and storing it in a vector database
# import os
# from dotenv import load_dotenv
# # from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_chroma import Chroma

# load_dotenv()

# HUGGING_FACE_API = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 


# # embedding_model = HuggingFaceEndpointEmbeddings(
# #     model="sentence-transformers/all-MiniLM-L6-v2",
# #     huggingfacehub_api_token=HUGGING_FACE_API
    
# #     )

# embedding_model = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001",
#     google_api_key=GOOGLE_API_KEY
# )

# vector_store = Chroma.from_texts(
#     texts = chunks,
#     embedding = embedding_model
# )

# # print("Embeddings stored in chroma db successfully")

# #Retrieving the data
# retriever=vector_store.as_retriever(search_kwargs={"k":3})

# #Initializing the llm

# from langchain_google_genai import ChatGoogleGenerativeAI

# # llm = ChatGoogleGenerativeAI(
# #     model="gemini-3.8-flash",
# #     google_api_key=GOOGLE_API_KEY,
# #     temperature=0.3
# # )

# from langchain_groq import ChatGroq

# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     groq_api_key=os.environ.get("GROQ_API_KEY"),
#     temperature=0.3
# )


# #Structure the prompt

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Keep the answer concise and simple.\n\n"
#     "Context:\n{context}"
# )

# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}")
# ])


# #Create the final RAG pipeline

# from langchain_classic.chains import create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# # Create a chain that formats text chunks into the prompt template
# question_answer_chain = create_stuff_documents_chain(llm, prompt)

# # Bind the retriever and Q&A chain together
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# #Run query!
# # user_query = "is dolo 650 present"

# user_query = input("Enter your query:")

# print("Pls wait sending query to the model")
# response = rag_chain.invoke({"input": user_query})

# print("\nModel response:")
# print(response["answer"])


# The above coding is working abs fine. jconveting that code into a streamlit applic.


import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load local .env variables (also works automatically with Streamlit Cloud Secrets)
load_dotenv()

st.set_page_config(page_title="Talk to PDF")
st.title("Talk to your PDF")

# Cache the RAG pipeline per uploaded file so it doesn't re-embed on every question
@st.cache_resource(show_spinner="Reading and indexing PDF...")
def build_rag_chain(file_bytes, file_name):
    # 1. Extract text from PDF
    import io
    document = PdfReader(io.BytesIO(file_bytes))
    fulltext = ""
    for page in document.pages:
        fulltext += page.extract_text() or ""

    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(fulltext)

    # 3. Create embeddings & vector store
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. Initialize Groq LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.3
    )

    # 5. Build prompt and RAG chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise and simple.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# --- UI Layout ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    rag_chain = build_rag_chain(uploaded_file.getvalue(), uploaded_file.name)
    st.success(f"Loaded **{uploaded_file.name}**! Ask a question below.")

    user_query = st.chat_input("Ask something about your PDF...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_query})
                st.write(response["answer"])
