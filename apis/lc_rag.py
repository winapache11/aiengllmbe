import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
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


if __name__ == "__main__":
    try:
        qa = check_schedule()
    except RuntimeError as e:
        print(str(e))
        raise SystemExit(1)

    query = "Summarize the content of the document."
    result = qa({"question": query, "chat_history": []})
    print("Response:", result['answer'])