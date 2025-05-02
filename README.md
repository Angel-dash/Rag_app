# 🎬 Movie Maniac Chatbot 🍿



A conversational AI agent designed to answer questions about movies. It intelligently utilizes a local movie database (built using Retrieval-Augmented Generation - RAG) from a PDF file and seamlessly falls back to real-time web search (via Tavily API) for comprehensive answers.

#Architecture
![image](https://github.com/user-attachments/assets/e91f7cd0-f120-43d3-a227-11ddbdfaa6d6)

![image](https://github.com/user-attachments/assets/1c146bc6-f0c9-45af-9dd0-151e40de99b9)


---

## ✨ Features

*   **Conversational Q&A:** Ask questions about movie plots, actors, directors, and more in natural language.
*   **Hybrid Search:**
    *   ⚡ **Fast Local Lookups:** Uses an internal vector database (ChromaDB, in-memory) built from `Data/Movie_description.pdf` for quick information retrieval via RAG.
    *   🌐 **Web Search Fallback:** If information isn't found locally, it searches the internet using the Tavily Search API.
*   **Context Aware:** Remembers the previous turns in the conversation for relevant follow-up questions.
*   **Source Transparency:** Indicates whether the answer came from the local database or an online search.
*   **Simple CLI Interface:** Easy to run and interact with directly from your terminal.

---
---
## 🛠️ Technology Stack

*   **Core Logic:** Python 3.9+
*   **LLM:** Google Gemini API (via `google-generativeai`)
*   **Agent Framework:** Google ADK (`google-cloud-aiplatform[adk]`)
*   **Web Search:** Tavily Search API (via `langchain-tavily`)
*   **RAG Backend:**
    *   `chromadb` (In-memory vector database)
    *   `sentence-transformers` (For text embeddings)
    *   `langchain-community` (For PDF loading/splitting)
*   **Configuration:** `python-dotenv` (For `.env` file management)
---
## 📋 Prerequisites

Before you begin, ensure you have the following:

1.  **Python:** Version 3.9 or higher is recommended.
2.  **API Keys:**
    *   🔑 **Google Gemini API Key:** Required for the core language model. Get one from [Google AI Studio](https://aistudio.google.com/).
    *   🔑 **Tavily Search API Key:** Required for the web search functionality. Get one from [Tavily AI](https://tavily.com/).

---

## 🚀 Getting Started

Follow these steps to set up and run the Movie Maniac Chatbot:

### 1. Clone the Repository

```bash
git clone https://github.com/Angel-dash/Rag_app/
cd Rag_app
```
### 2.Set Up API Keys
Create a file named .env in the root directory of the project.
Add your API keys to this file exactly like this:
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
TAVILY_API_KEY="YOUR_TAVILY"
3.  **Set Up Virtual Environment:**
    Using a virtual environment is highly recommended to manage dependencies. Choose one method:

    *   **Using `venv` (Python's built-in):**
        ```bash
        # Create environment (e.g., named 'env' or 'venv')
        python -m venv env

        # Activate environment
        # Linux/macOS:
        source env_API_KEY_HERE"
    ```
    *(Replace the placeholder text with your actual keys)*

*   **🔒 Security:** Add `.env` to your `.gitignore` file immediately to prevent accidentally committing your secret keys!
    ```bash
    echo ".env" >> .gitignore
    ```

### 3. Create & Activate Virtual Environment

Choose your preferred method:

*   **Using `venv` (Recommended):**
    ```bash
    # Create environment
    python -m venv venv

    # Activate environment
    # Linux/macOS:
    source venv/bin/activate
    # Windows (CMD/PowerShell):
    # venv\Scripts\activate
    ```

*   **Using `conda`:**
    ```bash
    # Create environment
    conda create -n movie_agent python=3.10 # Or your preferred Python version

    # Activate environment
    conda activate movie_agent
    ```

### 4. Install Dependencies

With your virtual environment activated, install the required packages:

```bash
/bin/activate
        # Windows (CMD/PowerShell):
        # env\Scripts\activate
        ```

    *   **Using `conda`:**
        ```bash
        # Create environment (e.g., named 'movie_agent')
        conda create -n movie_agent python=3.10 # Or your preferred Python 3.x version
        # Activate environment
        conda activate movie_agent
        ```

4.  **Install Dependencies:**
    With your virtual environment activated, install the required packages:
    ```bash
    pip install -r requirements.txt
```
▶️ Running the Application
Make sure your virtual environment is activated.
Navigate to the root directory of the project in your terminal.
Execute the main agent script:
```bash
python Agents/agents.py
```
