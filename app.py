# Load the pdf and extracting the text from it
from pypdf import PdfReader

#load the doc
document = PdfReader("pdfs/sample.pdf")

#Iterate through the pages of the pdf and extract the text
fulltext=""
for page in document.pages:
    fulltext += page.extract_text()

# print("The text of the ducment is shown below:\n")
# print(fulltext)



#Now chunking the text data

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  
    chunk_overlap=200
)

chunks = text_splitter.split_text(fulltext)

print(f"The size of chunks: {len(chunks)}")
# print("\n The first chunk is:",chunks[0])


#Creating embeddings for the text data and storing it in a vector database
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

load_dotenv()

HUGGING_FACE_API = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 


embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HUGGING_FACE_API
    
    )

vector_store = Chroma.from_texts(
    texts = chunks,
    embedding = embedding_model
)

# print("Embeddings stored in chroma db successfully")

#Retrieving the data
retriever=vector_store.as_retriever(search_kwargs={"k":3})

#Initializing the llm

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


#Structure the prompt

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Keep the answer concise and simple.\n\n"
    "Context:\n{context}"
)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


#Create the final RAG pipeline

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Create a chain that formats text chunks into the prompt template
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Bind the retriever and Q&A chain together
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


#Run query!
# user_query = "is dolo 650 present"

user_query = input("Enter your query:")

print("Pls wait sending query to the model")
response = rag_chain.invoke({"input": user_query})

print("\nModel response:")
print(response["answer"])
