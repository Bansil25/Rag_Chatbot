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


def load_index():
    """Load existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    return FAISS.load_local(
        INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True
    )


def build_rag_chain(vectorstore):
    """Return a runnable RAG chain given a loaded vectorstore."""
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    def doc_format(docs):
        parts = []
        for doc in docs:
            src = doc.metadata.get('source', 'unknown')
            parts.append(f'[Source: {src}]\n{doc.page_content}')
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


def ask(chain, question: str) -> str:
    """Invoke the chain with a question, return answer string."""
    return chain.invoke(question)
