# QuarkTutor MiddleWare
## 🎯 Project Overview

This project is a backend service for an intelligent and supportive physics teaching assistant designed for Senior High School (SHS) teachers in Ghana. The chatbot's primary goal is to help educators enhance their teaching methods, clarify complex physics concepts, and provide practical examples relevant to the Ghanaian context, all while aligning with the West African Senior School Certificate Examination (WASSCE) syllabus.

The application leverages a local Large Language Model (LLM), an in-memory chat history manager, and a curriculum search index to provide accurate and context-aware responses.

-----

## ✨ Features

  * **Custom LLM Integration:** Utilizes `llama-cpp` to run a local GGUF model for efficient and private language generation.
  * **Contextual Responses:** Incorporates a custom system prompt (`chat_utils.py`) to ensure all responses are tailored to the specific needs of Ghanaian physics teachers.
  * **Curriculum Search:** Indexes a curriculum document (PDF) using FAISS and `sentence-transformers` to provide relevant information as context to the LLM.
  * **Memory Management:** Implements a stateful memory system (`memory_manager.py`) that stores recent chat history and generates long-term summaries to maintain context across conversations.
  * **Robust Logging:** Uses a custom `TechSupportLogger` to create detailed, rotating log files for easy debugging and troubleshooting.
  * **FastAPI Backend:** Provides a high-performance, asynchronous API with clear endpoints for chat interactions and system health checks.

-----

## 🚀 Setup and Installation

Follow these steps to get the project up and running in your local environment.

### Prerequisites

  * Python 3.8 or higher
  * `pip` (Python package installer)

### Step 1: Clone the Repository and Navigate to the Directory

Assuming your project is in a Git repository, clone it and move into the project folder.

```bash
git clone https://github.com/mic-cyberkid/QuarkTutorServer
cd QuarkTutorServer
```

### Step 2: Set up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

Install the required Python packages. Create a **`requirements.txt`** file with the following content, then install them.

```bash
# requirements.txt
fastapi
uvicorn
llama-cpp-python
sqlalchemy
python-jose
passlib[bcrypt]
pydantic
pdfplumber
faiss-cpu
sentence-transformers
nltk
requests
```

```bash
pip install -r requirements.txt
```

### Step 4: Download the Language Model

You need to download a compatible GGUF model and place it in the **`models/`** directory. For this project, a Qwen model is used.

```bash
mkdir -p models
cd models
wget https://huggingface.co/reach-vb/Qwen3-0.6B-Q8_0-GGUF/resolve/main/qwen3-0.6b-q8_0.gguf
```

> **Note:** The `chat_llm_service.py` file is configured to look for `PhysicsChatBot.gguf` by default. You can either rename the downloaded file or update the `MODEL_FILENAME` variable in the `chat_llm_service.py` file.

### Step 5: Run the Server

To start the FastAPI server, run the following command from the root of your project directory:

```bash
uvicorn FullServer:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag is useful for development as it automatically restarts the server on code changes. You can remove it for production.

The API will be available at `http://localhost:8000`.

-----

## 🛠️ Project Structure

  * **`FullServer.py`**: The main FastAPI application file. It sets up the server, defines the routes (`/llm/chat`, `/llm/get_memory`), and initializes all the necessary services and logging.
  * **`chat_utils.py`**: Contains utility functions and the `SYSTEM_PROMPT` which defines the persona and instructions for the chatbot.
  * **`chat_llm_service.py`**: Encapsulates all the logic for interacting with the local LLM. It handles prompt formatting, model loading, and response generation.
  * **`curriculum_indexer.py`**: Manages the curriculum data. It reads a PDF, splits it into chunks, and creates a searchable FAISS index for retrieving relevant context.
  * **`memory_manager.py`**: Responsible for managing conversation history. It stores recent messages and creates a long-term summary to provide context for the LLM.
  * **`tech_support_logger.py`**: A custom logger class that provides a standardized way to log messages to files with rotation, which is crucial for debugging and technical support.

-----
