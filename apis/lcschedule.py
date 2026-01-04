import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))
api_keys = {
    'openai': os.getenv('OPENAI_API_KEY'),
    'google': os.getenv('GOOGLE_API_KEY'),
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),    
    'deepseek': os.getenv('DEEPSEEK_API_KEY'),
}

openai_api_key = api_keys['openai']


def check_schedule():
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Create a .env file at the project root with OPENAI_API_KEY=... and try again.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
    
    # Load and split documents
    sample_path = os.path.join(os.path.dirname(__file__), 'sample_docs', 'sample.txt')
    loader = TextLoader(sample_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Create conversational retrieval chain (using the current non-deprecated API)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return qa_chain

def upcoming_schedule(query: str):
    url = "https://365datascience.com/upcoming-courses"
    loader = WebBaseLoader(url)
    raw_documents = loader.load()
    print(f"Loaded {raw_documents} documents from {url}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    vectorstore = FAISS.from_documents(documents, embeddings)

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=openai_api_key),
        vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    result = qa_chain({"question": query})
    return result["answer"]


query = "What is the next course to be uploaded on the 365DataScience platform?"

if __name__ == "__main__":
    try:
        user_text = input("Enter your query (leave empty to paste multi-line and press Ctrl-D): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo input received. Exiting.")
        raise SystemExit(0)
    
    if user_text:
        print(upcoming_schedule(user_text))
    else:
        # Let text_summarizer prompt for multi-line input and summarize it.
        print(f"Response: {upcoming_schedule('No query provided.')}")