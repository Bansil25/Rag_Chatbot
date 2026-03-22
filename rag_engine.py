# rag_engine.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

INDEX_PATH = 'faiss_index'

'''
# USE TO UPLOAD SINGLE PDF ONLY
def build_index(pdf_path: str):
    """Load PDF → split → embed → save FAISS index to disk."""

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    return len(chunks)
'''

# USE TO UPLOAD MULTIPLE PDF'S
def build_index(pdf_paths: list):
    """Accept a list of PDF paths, index all of them."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs   = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    embeddings  = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return len(all_chunks)


def load_index():
    """Load existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    return FAISS.load_local(
        INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True
    )


def build_rag_chain(vectorstore):

# Return a runnable RAG chain given a loaded vectorstore.
   # retriever = vectorstore.as_retriever(search_kwargs={'k': 3}) # basic top k only

    '''
    MMR (Maximal Marginal Relevance) returns more diverse chunks — better answers for complex questions.
    fetch_k=10 means FAISS fetches 10 candidates first, then MMR picks the 3 most diverse ones from those 10.
    Better coverage of the document, less repetitive context.
    '''
     retriever = vectorstore.as_retriever(
     search_type="mmr",
     search_kwargs={"k": 3, "fetch_k": 10}
        )


   # ADD SOURCE ONLY
    def doc_format(docs):
        parts = []
        for doc in docs:
            src = doc.metadata.get('source', 'unknown')
            parts.append(f'[Source: {src}]\n{doc.page_content}')
        return "\n\n".join(parts)
    '''
    # ADD SOURCE AND PAGE NUMBERS AS WELL
    def doc_format(docs):
    parts = []
    for doc in docs:
        src  = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', '?')
        parts.append(f'[Source: {src}, Page: {page + 1}]\n{doc.page_content}')
    return "\n\n".join(parts)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions
based ONLY on the provided document context.
If the answer is not in the context, say:
'I could not find that information in the uploaded document.'
Always mention which page or source the answer came from.

Context:
{context}"""),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    chain = (
        {
            "context":  retriever | doc_format,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
'''
# ADDED CHAT MEMORY (CHATBOT REMEMBERS PREVIOUS QUESTION IN THE SAME SEASSION)

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    def doc_format(docs):
        parts = []
        for doc in docs:
            src  = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', '?')
            parts.append(f'[Source: {src}, Page: {page+1}]\n{doc.page_content}')
        return "\n\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions
based ONLY on the provided document context.
If the answer is not in the context, say:
'I could not find that information in the uploaded document.'
Always mention which page or source the answer came from.
Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    chain = (
        {
            "context":      retriever | doc_format,
            "question":     RunnablePassthrough(),
            "chat_history": lambda _: st.session_state.get("chat_history", [])
        }
        | prompt | llm | StrOutputParser()
    )
    return chain

def ask(chain, question: str) -> str:
    """Invoke the chain with a question, return answer string."""
    return chain.invoke(question)
