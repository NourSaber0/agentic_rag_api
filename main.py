import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# Fail fast: If the server boots without these, it crashes immediately.
if not all([GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise ValueError("Missing critical environment variables! Check your .env file or Cloud Secrets.")

app = FastAPI(
    title="Agentic RAG API",
    description="An API that uses a LangGraph Agent to answer queries based on uploaded docs, a DB, or live web data."
)

# ---------------------------------------------------------
# 1. Initialize Embeddings & Vector Store
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
if not qdrant_client.collection_exists("knowledge_base"):
    qdrant_client.create_collection(
        collection_name="knowledge_base",
        # all-mpnet-base-v2 outputs exactly 768 dimensions
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="knowledge_base",
    embedding=embeddings,
)

# ---------------------------------------------------------
# 2. Define the Agent Tools
# ---------------------------------------------------------
@tool
def search_knowledge_base(query: str) -> str:
    """Use this tool to search the uploaded PDF documents for information.
    Input should be a specific search query."""
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "No uploaded documents found matching the query."
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def check_account_status(user_id: str) -> str:
    """Use this tool to check the status of a user's account.
    Input must be the user's ID (e.g., 'user_123')."""
    mock_db = {
        "user_123": "Account Active. Subscription: Premium.",
        "guest": "Account Inactive. Subscription: None."
    }
    return mock_db.get(user_id, "User ID not found in database.")

web_search_tool = DuckDuckGoSearchRun(
    name="search_the_web",
    description="Use this tool to search the live internet for recent events or information not found in the documents."
)

tools = [search_knowledge_base, check_account_status, web_search_tool]

# ---------------------------------------------------------
# 3. Build the Agentic Orchestrator (LangGraph V1.0)
# ---------------------------------------------------------
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY)

system_prompt = (
    "You are a helpful customer support agent. "
    "You have access to tools to look up user accounts, search internal manuals, and search the web. "
    "Use them when necessary to provide accurate answers."
)

memory = MemorySaver()

# Create the graph-based agent. 
# We pass the memory saver, and the LLM will automatically track history per user.
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory
)

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    user_id: str = "guest"

class ChatResponse(BaseModel):
    answer: str
    source: str
    tools_used: list[str] = []

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
        # --- NEW CODE: Use a proper temporary file that PyPDFLoader can read safely ---
    try:
        # Create a temporary file that deletes itself when we are done
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            # Write the uploaded bytes to the temp file
            content = await file.read()
            tmp.write(content)
            temp_file_path = tmp.name

        # Now PyPDFLoader can safely read it from the hard drive
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vector_store.add_documents(documents=chunks)
        
        return {"filename": file.filename, "status": "Success", "chunks_created": len(chunks)}
    
    except Exception as e:
        # If it fails, give us the real error message instead of a generic 500 error!
        return {"filename": file.filename, "status": "Error", "message": str(e)}
        
    finally:
        # Clean up the temporary file so we don't fill up the hard drive
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    try:
        # LangGraph relies on a 'thread_id' to differentiate users and keep their memories separate.
        config = {"configurable": {"thread_id": request.user_id}}
        
        # We pass the user's message into the agent. 
        # The checkpointer automatically pulls their past messages and appends this new one!
        response = agent.invoke(
            {"messages": [("user", request.query)]}, 
            config=config
        )
        
        # The agent returns a dictionary with the full message history. The last message is its final answer.
        final_answer = response["messages"][-1].content
        used_tool = []
        for msg in response["messages"]:
            # Check if this message is the actual RESULT of a tool being run
            if msg.type == "tool":
                used_tool.append(msg.name)

                used_tool = list(dict.fromkeys(used_tool))  # Remove duplicates

        return ChatResponse(answer=final_answer, source="LangGraph Orchestrator", tools_used=used_tool)
    
    except Exception as e:
        return ChatResponse(answer=f"Agent encountered an error: {str(e)}", source="Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
