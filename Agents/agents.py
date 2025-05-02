import os 
from google.adk.agents import LlmAgent, Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google import genai 
from google.genai import types
from langchain_tavily import TavilySearch
from pathlib import Path 
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv()
from Rag.rag_pipeline import initialize_rag_database,query_rag_database

PDF_DIR = "Data/Movie_description.pdf"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.Client(api_key=GOOGLE_API_KEY)
    print("Gemini API Key Configured via genai.configure()")
else:
    print("Warning: GOOGLE_API_KEY environment variable not set.")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
tool = TavilySearch(max_result = 3, topic = "general")


def query_movie_database_tool(query: str) -> dict:
    """
    Queries the pre-loaded movie database for relevant documents using vector similarity.
    Does not generate the final answer, only retrieves context.
    """
    print(f"\n[RAG TOOL CALLED]: Querying local movie database (RAG) for: '{query}'")
    try:
        retrieved_docs = query_rag_database(query, n_results=3)

        if retrieved_docs:
             print(f"[RAG TOOL RESULT]: RAG tool found {len(retrieved_docs)} relevant documents.")
             return {"retrieved_documents": retrieved_docs, "status": "success"}
        else:
             print("[RAG TOOL RESULT]: RAG tool found no relevant documents.")
             return {"retrieved_documents": [], "status": "success", "message": "No relevant documents found in the database."}
    except Exception as e:
        print(f"[RAG TOOL ERROR]: RAG query failed: {e}")
        print(f"Error in query_movie_database_tool: {e}", exc_info=True)
        return {"retrieved_documents": [], "status": "error", "message": str(e)}
    

def search_tool(user_query:str)->str:
    """Use Tavily Search to search the internet"""
    print("\n[SEARCH TOOL CALLED]: Attempting to search the internet (Tavily)...")
    search_result = tool.invoke({"query":user_query})
    print("\n[SEARCH TOOL CALLED]: Internt search completed (Tavily)...")
    return search_result

self_description_agent = LlmAgent(
    name = "SelfDescriptionAgent", 
    model = "gemini-2.0-flash", 
    description = "Describe the capabilites of the system", 
    instruction = """
    You are a part of the system which contain an agent called "Movie-Maniac" which answers question based on movies,
    so your job is to answer about the capabilites of the system, what it can do and how it works.

    #Examples
    User: What can you do ?
    Agent: I can answer question realted to the movies.

    Here are the list of your capabilites,
    1. I can use a built-in movie database to search for movie facts, descriptions, or actors.
    2. If I cannot find the information in the database, I can search the internet using a search tool.
    3. I can maintain context during our conversation, allowing for follow-up questions.
    """,
    output_key = "self_descriptions_answers"
)

movie_agent = LlmAgent(
    name = "MovieSearchAgent", 
    model = "gemini-2.0-flash", 
    description = "Search internal movie database using RAG and if needed use the search tool to search the internet.",
    instruction = """
    You are the primary agent for answering movie-related questions within the Movie-Maniac system.
    **Crucially, you must use the entire conversation history to understand the user's current query, especially for follow-up questions.**

    Follow these steps:

    1.  **Analyze the Conversation History and Current Query:** Understand the user's intent, taking into account what has been discussed previously. If the current query is a follow-up (e.g., "Who directed it?", "Tell me more about the actor?"), refer back to the prior turns to identify the movie, actor, or topic being referenced.

    2.  **Prioritize Internal Database:** **First, use the `query_movie_database_tool`** with a query formulated based on your understanding of the conversation history and the current user input.
        *   Example: If the user asked "Tell me about The Shawshank Redemption" and then "Who starred in it?", call `query_movie_database_tool` with a query like "actors in The Shawshank Redemption".

    3.  **Analyze RAG Tool Result:**
        *   If the `query_movie_database_tool` returns `status: "success"` and has an `answer` field containing relevant information, use this answer to respond to the user.
        *   If the `status` is "success" but the `answer` is empty or the message indicates "No relevant documents found...", proceed to step 4.
        *   If the `status` is "error", acknowledge the issue and proceed to step 4.

    4.  **Use Search Tool as Fallback (if RAG fails):** If you could not find the information using `query_movie_database_tool`, **use the `search_tool`** to search the internet. Formulate a precise search query based on the user's request and the conversation history.
        *   Example: If RAG failed for "actors in The Shawshank Redemption", call `search_tool` with a query like "actors in The Shawshank Redemption".

    5.  **Synthesize Final Answer:** Combine information from the tool results (preferably RAG, then Search) to formulate a comprehensive, natural language answer to the user's query.
        *   **Important:** Address the user's question directly, incorporating information found. Do NOT simply output the raw results from the tools. Explain what you found.
        *   If you couldn't find information from either source, politely state that you couldn't find an answer to their question.

    6.  **Maintain Conversational Flow:** Ensure your response is contextually aware and flows naturally with the preceding turns.

    """,
    tools = [query_movie_database_tool, search_tool],
    output_key = "rag_agent_answers"
)

router_agent = LlmAgent(
    name = "MovieManicRouter",
    model = "gemini-2.0-flash",
    description = "The main router for the MovieManiac system",
    instruction="""
    You are the MovieManiac Router. Your job is to analyze the user's query and delegate the task to the appropriate sub-agent.
    - If the user's query is a greeting (e.g., "Hi", "Hello") or asks about your capabilities (e.g., "What can you do?", "Tell me about yourself"), delegate to the 'SelfDescriptionAgent'.
    - If the user's query is about movies, actors, descriptions, plots, or any other movie-related topic, delegate to the 'MovieSearchAgent'.
    - Do not try to answer the user's question directly. Your only task is to route the request to the correct sub-agent.
    """,
    sub_agents=[self_description_agent, movie_agent]
)

APP_NAME = "movie_rag_agent"
USER_ID = "user123"
SESSION_ID = "session456"

session_service = InMemorySessionService()
session = session_service.create_session(app_name = APP_NAME, user_id = USER_ID, session_id =SESSION_ID)
runner = Runner(agent= router_agent, app_name = APP_NAME,session_service= session_service)


def chat_with_agent(query):
    content = types.Content(role= 'user', parts = [types.Part(text= query)])
    events = runner.run(user_id = USER_ID,session_id = SESSION_ID, new_message= content )
    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            return final_response
print("Attempting to initialize RAG database...")
if PDF_DIR and initialize_rag_database(PDF_DIR):
    print("RAG Database initialized successfully from agents.py.")
else:
    print("WARNING: RAG Database initialization failed or PDF_DIR not set. RAG tool will not work.")
def start_conversation():
    print("Movie-Maniac Bot: Hello! I'm Movie-Maniac. Ask me anything about movies!")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Movie-Maniac Bot: Thanks for chatting! Goodbye!")
            break
    
        response = chat_with_agent(user_input)
        print(f"Movie-Maniac Bot: {response}")

# Start the conversation
if __name__ == "__main__":
    start_conversation()