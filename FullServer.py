# Server Code (FullServer.py)
from fastapi import FastAPI, APIRouter, Body, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from memory_manager import MemoryManager
from chat_llm_service import ChatLLMService
from passlib.context import CryptContext
from sqlalchemy.dialects.sqlite import BLOB as SQLiteUUID
from sqlalchemy.types import CHAR
from curriculum_indexer import CurriculumIndexer
from jose import JWTError, jwt
from passlib.hash import bcrypt
from datetime import datetime, timedelta
from pydantic import BaseModel
import urllib.request as downloader
from llama_cpp import Llama
import shutil
import json
import uuid
import sqlite3
import requests 
import os
import logging # Keep this for logging levels
from contextlib import asynccontextmanager # For FastAPI lifespan events
from tech_support_logger import TechSupportLogger # Import our custom logger

# --- Initialize our custom logger for the entire application ---
app_logger_instance = TechSupportLogger(
    log_file_name="middleware.log",
    log_dir="data/logs", # All logs will go here
    level=logging.DEBUG, # Set to logging.INFO for production, logging.DEBUG for verbose troubleshooting
    max_bytes=50 * 1024 * 1024, # 50 MB per log file
    backup_count=10, # Keep 10 backup log files
    console_output=True # Set to False for true background service if not debugging
)
logger = app_logger_instance.get_logger() # Get the logger instance to use

# --- CONFIGURATION ---
DB_URL = "sqlite:///data/chatbot.db"
SECRET_KEY = "super-secret-key-replace-with-env-var-in-prod" # IMPORTANT: We for change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # One day 

# LLM Model Configuration (for bundled PyInstaller app)
# We for make sure sey 'models/PhysicsChatBot.gguf' exists relative to our executable.
MODEL_DIR = "models"
MODEL_FILENAME = "PhysicsChatbot.gguf" # Qwen3-0.6b GGUF 
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Ensure wanna model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# GREETINGS for special handling in chat (e.g., no curriculum search) : 
GREETINGS = { # Using a set for faster lookup
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "morning", "afternoon", "evening", "hola", "salutations", "greetings", "thanks","tanx","thank you",
    "no", "yes" 
}

# --- DATABASE SETUP ---
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global store for MemoryManager instances (per user, per conversation)
# This allows keeping memory loaded for active conversations
memory_store = {}

# --- LLM and Indexer Initialization ---
# Initialize LLM service and curriculum indexer once at startup
try:
    llm_service = ChatLLMService(model_path=MODEL_PATH)
    logger.info(f"ChatLLMService initialized with model: {MODEL_PATH}")
except Exception as e:
    logger.critical(f"Failed to initialize ChatLLMService: {e}", exc_info=True)
    # Depending on severity, you might want to exit or raise an error here
    # For now, we'll let the app try to start, but LLM functionality will fail.
    llm_service = None # Set to None if initialization fails
    raise e

try:
    curriculum_indexer = CurriculumIndexer(
        pdf_folder="pdfs" # Ensure our 'pdfs' directory exists and contains PDFs
    )
    logger.info("CurriculumIndexer initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize CurriculumIndexer: {e}", exc_info=True)
    curriculum_indexer = None # Set to None if initialization fails

# --- FastAPI Lifespan Events ---
# This ensures graceful startup and shutdown, especially for saving MemoryManager states.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("FastAPI application startup initiated.")

    # Check for LLM model file on startup (if not already handled by PyInstaller bundling)
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"LLM model not found at {MODEL_PATH}. Attempting to download if URL is configured.")
       
    yield # Application is ready to receive requests

    # Shutdown logic
    logger.info("FastAPI application shutdown initiated. Saving all active memory managers...")
    for key, mem_manager in list(memory_store.items()): # Iterate over a copy to allow modification
        try:
            logger.info(f"Saving memory for conversation: {key}")
            mem_manager._safe_save() # Call the safe save method
            del memory_store[key] # Remove from store after saving
        except Exception as e:
            logger.error(f"Failed to save memory for {key} during shutdown: {e}", exc_info=True)
    logger.info("All active memory managers saved. FastAPI application shutdown complete.")


# --- APP SETUP ---
app = FastAPI(lifespan=lifespan)
chat_router = APIRouter()
# Initialize database schema
def init_db():
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized.")

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # IMPORTANT: Restrict this in production to our JavaFX app's origin [ Not really needed tho]
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/llm/token")

# --- UTILS ---
def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- MODELS ---
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    chats = relationship("Chat", back_populates="user")

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="chats")
    conversation_id = Column(String(36), index=True) # For compatibility across DBs


# --- AUTHENTICATION FUNCTIONS ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Hashes a plain password using bcrypt."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- AUTH ROUTES ---
@chat_router.post("/register", summary="Register a new user")
def register_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Attempting to register user: {form_data.username}")
    try:
        if db.query(User).filter_by(username=form_data.username).first():
            logger.warning(f"Registration failed: Username '{form_data.username}' already exists.")
            raise HTTPException(status_code=400, detail="Username already exists")
        hashed = get_password_hash(form_data.password)
        user = User(username=form_data.username, hashed_password=hashed)
        db.add(user)
        db.commit()
        db.refresh(user) # Refresh to get the user ID
        logger.info(f"User '{form_data.username}' registered successfully with ID: {user.id}")
        return {"msg": "User registered successfully"}
    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Error during user registration for '{form_data.username}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during registration")

@chat_router.post("/token", summary="Authenticate user and get access token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Attempting to log in user: {form_data.username}")
    try:
        user = db.query(User).filter_by(username=form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            logger.warning(f"Login failed for user '{form_data.username}': Invalid credentials.")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_access_token({"sub": user.username})
        logger.info(f"User '{form_data.username}' logged in successfully.")
        return {"access_token": token, "token_type": "bearer", "username": user.username}
    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Error during user login for '{form_data.username}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during login")

@chat_router.post("/update-password", summary="Update user password")
def update_password(
    token: str = Depends(oauth2_scheme),
    old_password: str = Body(..., embed=True), # Use embed=True for single field in body
    new_password: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    logger.info(f"Attempting to update password for user via token.")
    try:
        # Verify token validity and expiration date
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": True}
        )
        username = payload.get("sub")

        if not username:
            logger.warning("Password update failed: Invalid token payload (no username).")
            raise HTTPException(status_code=401, detail="Invalid token or expired")

        user = db.query(User).filter_by(username=username).first()
        if not user:
            logger.warning(f"Password update failed for user '{username}': User not found.")
            raise HTTPException(status_code=404, detail="User not found")

        # Input validation
        if not old_password or not new_password:
            logger.warning(f"Password update failed for user '{username}': Missing input parameters.")
            raise HTTPException(status_code=400, detail="Missing old_password or new_password")

        # Validate new password length and format
        if len(new_password) < 8:
            logger.warning(f"Password update failed for user '{username}': New password too short.")
            raise HTTPException(
                status_code=400, detail="New password must be at least 8 characters long"
            )

        # Check if the provided old password matches the hashed password in the database
        if not verify_password(old_password, user.hashed_password):
            logger.warning(f"Password update failed for user '{username}': Invalid old password.")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid old password")

        # Hash the new password and update in the database
        user.hashed_password = get_password_hash(new_password)
        db.add(user) # Add the modified user object back to the session
        db.commit() # COMMIT THE CHANGES TO THE DATABASE
        db.refresh(user) # Refresh the user object to reflect committed changes
        logger.info(f"Password updated successfully for user: {username}")
        return {"msg": "Password updated successfully"}

    except JWTError as e:
        logger.error(f"JWT error during password update: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Unhandled error during password update: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during password update")

@chat_router.post("/manual-update-password", summary="Update password using username and password")
def manual_update_password(
    username: str = Body(..., embed=True),
    old_password: str = Body(..., embed=True),
    new_password: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    logger.info(f"Attempting manual password update for user: {username}")
    try:
        user = db.query(User).filter_by(username=username).first()
        if not user:
            logger.warning(f"Manual password update failed: User '{username}' not found.")
            raise HTTPException(status_code=404, detail="User not found")
            return {"msg": "User not found!"} # mabr3 continue tomorrow

        # Input validation
        if not new_password:
            logger.warning(f"Manual password update failed for user '{username}': Missing input.")
            raise HTTPException(status_code=400, detail="Missing new_password")

        
        # Update and commit
        user.hashed_password = get_password_hash(new_password)
        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"Password manually updated successfully for user: {username}")
        return {"msg": "Password updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error during manual password update: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during password update")


# --- CHAT ROUTES ---
@chat_router.post("/chat", summary="Send a message to the chatbot")
async def chat( # Made async to align with StreamingResponse, though core logic is sync
    conversation_id: str = Body(None),
    message: str = Body(...),
    new_chat: bool = Body(False),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during chat request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"Chat request failed: User '{username}' not found.")
        raise HTTPException(status_code=404, detail="User not found")

    if new_chat:
        if conversation_id:
            logger.warning(f"Chat request for user '{username}': Cannot provide both new_chat and conversation_id.")
            raise HTTPException(status_code=400, detail="Cannot provide both new_chat and conversation_id")
        conversation_id = str(uuid.uuid4())
        logger.info(f"User '{username}' started a new conversation: {conversation_id}")
    elif not conversation_id:
        logger.warning(f"Chat request for user '{username}': conversation_id not specified for existing chat.")
        raise HTTPException(status_code=400, detail="conversation_id not specified")
    elif not db.query(Chat).filter_by(conversation_id=conversation_id).first():
        logger.warning(f"Chat request for user '{username}': Invalid Conversation ID '{conversation_id}'.")
        raise HTTPException(status_code=400, detail="Invalid Conversation ID")

    key = f"{username}:{conversation_id}"
    memory_dir = os.path.join("data", "memory", username, conversation_id)
    os.makedirs(memory_dir, exist_ok=True) # Ensure memory directory exists
    index_path = os.path.join(memory_dir, "faiss.index")
    memory_path = os.path.join(memory_dir, "memory.pkl")

    # Load or create MemoryManager for this conversation
    if key not in memory_store:
        logger.info(f"Loading/Creating MemoryManager for conversation: {key}")
        memory = MemoryManager(index_path=index_path, memory_path=memory_path)
        # If memory files don't exist, populate recent history from DB
        if not os.path.exists(index_path) or not os.path.exists(memory_path):
            history = db.query(Chat).filter_by(user_id=user.id, conversation_id=conversation_id).order_by(Chat.timestamp.asc()).all()
            for h in history:
                memory.add_message(h.role, h.content)
            logger.info(f"Populated MemoryManager from DB for {key}. Messages loaded: {len(history)}")
        memory_store[key] = memory
    else:
        logger.debug(f"Using existing MemoryManager for conversation: {key}")

    memory = memory_store[key]
    memory.add_message("user", message)
    logger.debug(f"User message added to memory for {key}.")

    # Get context and relevant curriculum chunks
    context = memory.get_context(message)
    relevant_chunks = []
    if message.lower() not in GREETINGS and curriculum_indexer: # Only search if not a greeting and indexer is available
        try:
            relevant_chunks = curriculum_indexer.search(message)
            logger.debug(f"Found {len(relevant_chunks)} relevant curriculum chunks for '{message[:50]}...'.")
        except Exception as e:
            logger.error(f"Error searching curriculum index for '{message[:50]}...': {e}", exc_info=True)
            relevant_chunks = [] # Ensure it's an empty list on error
    elif not curriculum_indexer:
        logger.warning("CurriculumIndexer not initialized, skipping curriculum search.")
    else:
        logger.debug(f"Message '{message[:20]}...' is a greeting, skipping curriculum search.")

    if not llm_service:
        logger.error("LLM service is not initialized. Cannot generate response.")
        raise HTTPException(status_code=503, detail="Chatbot service unavailable.")

    formatted_prompt = llm_service.format_prompt(message, context, relevant_chunks)
    logger.debug(f"Formatted prompt generated for LLM. Prompt length: {len(formatted_prompt)} characters.")

    async def stream_and_store():
        assistant_reply = ""
        # Early yield to initiate stream on client side
        yield json.dumps({"response": ""}) + "\n"
        logger.debug("Initiated streaming response.")

        try:
            # Start streaming tokens from the model
            # Note: llama_cpp's stream is synchronous, so we're still blocking
            # the FastAPI worker thread here. For high concurrency, consider
            # offloading LLM calls to a separate process/thread pool.
            for token in llm_service.generate_response_stream(formatted_prompt):
                assistant_reply += token
                # logger.debug(f"Streaming token: {repr(token)}") # Too verbose for INFO level
                yield json.dumps({"response": token}) + "\n"
        except Exception as e:
            logger.error(f"LLM streaming error for conversation {key}: {e}", exc_info=True)
            yield json.dumps({"error": "Streaming failed"}) + "\n"
            return # Early exit on error if streaming fails completely

        # Post-streaming processing
        logger.info(f"[+] RAW LLM response for {key}: {assistant_reply[:500]}...") # Log first 500 chars
        try:
            memory.add_message("assistant", assistant_reply)
            logger.debug(f"Assistant reply added to memory for {key}.")

            # Save chat history to database
            db.add(Chat(user_id=user.id, conversation_id=conversation_id, role="user", content=message))
            db.add(Chat(user_id=user.id, conversation_id=conversation_id, role="assistant", content=assistant_reply))
            db.commit()
            logger.info(f"Chat messages saved to database for conversation {conversation_id}.")
        except Exception as e:
            logger.error(f"Failed to save chat to DB for conversation {conversation_id}: {e}", exc_info=True)
            # Do not re-raise, as streaming might already be in progress, but inform client.
            yield json.dumps({"error": "Failed to save chat history"}) + "\n"


    return StreamingResponse(stream_and_store(), media_type="application/json", headers={"X-Conversation-ID": conversation_id})


@chat_router.post("/conversion", summary="Explain unit conversions or physics concepts")
async def get_conversion(message: str = Body(...)):
    logger.info(f"Received conversion/explanation request: {message[:100]}...")
    try:
        if not llm_service:
            logger.error("LLM service is not initialized. Cannot generate conversion explanation.")
            raise HTTPException(status_code=503, detail="Chatbot service unavailable.")

        # For conversion, we might not need memory or curriculum chunks, just direct LLM generation
        formatted_prompt = llm_service.format_prompt(message, [], [])
        response = llm_service.generate_response(formatted_prompt)
        logger.info(f"Generated conversion response for: {message[:50]}...")
        return {"response": response}
    except RuntimeError as e:
        logger.error(f"Runtime error during conversion explanation for '{message[:100]}...': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unhandled error during conversion explanation for '{message[:100]}...': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during conversion explanation")


# --- CONVERSATIONS ROUTES ---
@chat_router.get("/conversations", summary="List all user conversations")
async def list_conversations(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during list_conversations request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"List conversations failed: User '{username}' not found.")
        raise HTTPException(status_code=404, detail="User not found")

    conversation_ids_query = (
        db.query(Chat.conversation_id)
        .filter_by(user_id=user.id)
        .distinct()
        .all()
    )
    conversation_ids = [conv_id_tuple[0] for conv_id_tuple in conversation_ids_query]
    logger.info(f"User '{username}' requested conversations. Found {len(conversation_ids)} distinct conversations.")

    conversations = []
    for conv_id in conversation_ids:
        # Fetch the first user message and first assistant message for header/description
        first_user_msg = (
            db.query(Chat)
            .filter_by(user_id=user.id, conversation_id=conv_id, role="user")
            .order_by(Chat.timestamp.asc())
            .first()
        )
        first_assistant_msg = (
            db.query(Chat)
            .filter_by(user_id=user.id, conversation_id=conv_id, role="assistant")
            .order_by(Chat.timestamp.asc())
            .first()
        )

        header = first_user_msg.content if first_user_msg else "No user message"
        description = first_assistant_msg.content if first_assistant_msg else "No assistant message"

        conversations.append({
            "conversation_id": conv_id,
            "conversation_header": " ".join(header.split()[:20]), # Truncate for display
            "conversation_desc": " ".join(description.split()[:20]) # Truncate for display
        })
    return {"messages": conversations, "total": len(conversations)}

@chat_router.get("/conversations/{conversation_id}", summary="Get messages for a specific conversation")
async def get_conversation(conversation_id: str, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    logger.info(f"Fetching conversation '{conversation_id}' for user via token.")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during get_conversation request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"Get conversation failed: User '{username}' not found.")
        raise HTTPException(status_code=404, detail="User not found")

    chats = db.query(Chat).filter_by(user_id=user.id, conversation_id=conversation_id).order_by(Chat.timestamp).all()
    if not chats:
        logger.warning(f"Conversation '{conversation_id}' not found for user '{username}'.")
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msgs = [chat.content for chat in chats if chat.role == "user"]
    bot_msgs = [chat.content for chat in chats if chat.role == "assistant"]

    logger.info(f"Retrieved {len(user_msgs)} user messages and {len(bot_msgs)} bot messages for conversation '{conversation_id}'.")
    return {
        "conversation_id": conversation_id,
        "conversations": {
            "userMessages": user_msgs,
            "botMessages": bot_msgs
        }
    }

@chat_router.delete("/conversations/{conversation_id}", summary="Delete a conversation")
async def delete_conversation(conversation_id: str, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    logger.info(f"Attempting to delete conversation '{conversation_id}' for user via token.")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during delete_conversation request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"Delete conversation failed: User '{username}' not found.")
        raise HTTPException(status_code=404, detail="User not found")

    chats = db.query(Chat).filter_by(user_id=user.id, conversation_id=conversation_id).all()
    if not chats:
        logger.warning(f"Conversation '{conversation_id}' not found for user '{username}'. No deletion performed.")
        raise HTTPException(status_code=404, detail="Conversation not found")

    for chat in chats:
        db.delete(chat)
    db.commit()
    logger.info(f"Conversation '{conversation_id}' and its messages deleted from DB for user '{username}'.")

    # Securely delete memory files and remove from in-memory store
    key = f"{username}:{conversation_id}"
    memory_dir = os.path.join("data", "memory", username, conversation_id)
    if os.path.exists(memory_dir):
        try:
            shutil.rmtree(memory_dir)
            logger.info(f"Successfully deleted memory directory: {memory_dir}")
        except Exception as e:
            logger.error(f"Failed to delete memory directory {memory_dir}: {e}", exc_info=True)
    else:
        logger.warning(f"Memory directory not found for deletion: {memory_dir}")

    if key in memory_store:
        del memory_store[key]
        logger.info(f"Removed MemoryManager from in-memory store for {key}.")

    return {"status": "deleted", "conversation_id": conversation_id}


@chat_router.get("/debug/memory", summary="Get debug information about conversation memory")
async def debug_memory(conversation_id: str = Query(...), token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    logger.info(f"Debug memory request for conversation '{conversation_id}' for user via token.")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT error during debug_memory request: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        logger.warning(f"Debug memory failed: User '{username}' not found.")
        raise HTTPException(status_code=404, detail="User not found")

    key = f"{username}:{conversation_id}"
    memory_dir = os.path.join("data", "memory", username, conversation_id)
    index_path = os.path.join(memory_dir, "faiss.index")
    memory_path = os.path.join(memory_dir, "memory.pkl")

    if key not in memory_store:
        logger.info(f"Memory for {key} not in active store. Attempting to load from disk for debug.")
        if not os.path.exists(index_path) or not os.path.exists(memory_path):
            logger.warning(f"No memory files found for conversation '{conversation_id}'.")
            raise HTTPException(status_code=404, detail="No memory found for this conversation.")
        memory = MemoryManager(index_path=index_path, memory_path=memory_path)
        memory_store[key] = memory # Add to store for future access if it's debugged
        logger.info(f"MemoryManager for {key} loaded from disk for debug.")
    else:
        logger.debug(f"Memory for {key} found in active store for debug.")

    memory = memory_store[key]

    return {
        "recent_history_count": len(memory.recent_history),
        "recent_history_preview": [f"{m['role']}: {m['content'][:50]}..." for m in memory.recent_history],
        "summary": memory.summarize_old(),
        "long_term_count": len(memory.long_term_memory),
        "vector_index_size": memory.index.ntotal if memory.index else 0
    }


# --- ROUTE REGISTRATION ---
app.include_router(chat_router, prefix="/llm")

@app.get("/", summary="Health check endpoint")
def root():
    logger.info("Health check request received.")
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
