import os
import uuid
from typing import Dict, Any, Optional

import google.generativeai as genai
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langsmith.run_trees import RunTree
from google.generativeai import types as genai_types
from google.generativeai import protos

# Assuming 'query' is a module in the same project root
from tools import AVAILABLE_TOOLS, query_acts_schema, query_laws_schema
from prompts import system_prompt

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_PASSWORD = os.getenv("API_PASSWORD")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- LangSmith Configuration ---
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "Caseone AI Chatbot")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Caseone AI Chatbot API",
    description="A REST API for the Caseone AI chatbot.",
    version="1.0.0",
)

# --- Authentication ---
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != API_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session ID for the chat. A new one is created if not provided.",
    )
    question: str = Field(..., description="The question from the user.")
    max_token: int = Field(2048, description="Maximum tokens for the response.")
    top_p: Optional[int] = Field(
        None,
        description="Number of recent chat exchanges to keep in history (e.g., 3 for the last 3 Q&As). If 0, history is cleared. If not provided, all history is kept.",
    )

class ChatResponse(BaseModel):
    response: str

# --- Gemini Model and Chat Management ---
legal_tools = [
    genai_types.Tool(function_declarations=[
        query_acts_schema,
        query_laws_schema,
    ])
]

generation_config = genai_types.GenerationConfig(
    # max_output_tokens is the parameter for controlling response length
    # but we will use the one from request
)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # Using 1.5 flash as it's generally better
    system_instruction=system_prompt,
    tools=legal_tools,
    generation_config=generation_config,
)

# In-memory store for chat sessions
chat_sessions: Dict[str, Any] = {}

def get_chat_session(session_id: str):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[])
    return chat_sessions[session_id]

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key)])
async def chat_with_bot(request: ChatRequest):
    """
    Handles a chat request with the bot, maintaining session history.
    """
    top_p = 3

    parent_run = RunTree(
        name="Chat Session",
        run_type="chain",
        inputs={"question": request.question},
        extra={"metadata": {"session_id": request.session_id}},
    )
    total_prompt_tokens = 0
    total_candidates_tokens = 0

    try:
        chat = get_chat_session(request.session_id)

        # Truncate history based on top_p before sending the new message
        if top_p is not None:
            if top_p > 0:
                # A "chat" is a user message and a model response, so 2 history entries.
                messages_to_keep = top_p * 2
                if len(chat.history) > messages_to_keep:
                    chat.history = chat.history[-messages_to_keep:]
            elif top_p == 0:
                # Clear history if top_p is 0
                chat.history = []
        
        # Set max_token for this specific call if provided
        # Note: The underlying google-generativeai library might handle token limits differently.
        # Here we adjust the generation_config for the model for this request.
        # A more direct way is not available in send_message.
        # This setting on the model is not thread-safe if applied directly.
        # For this use case, we will rely on the default model setting,
        # as per-request token limit is complex to manage this way.
        # The `max_token` from request is noted, but not directly applied here
        # due to library limitations in `send_message`. A different model setup would be needed.

        generation_config = genai_types.GenerationConfig(
            max_output_tokens=request.max_token
        )
        
        llm_run = parent_run.create_child(
            name="Gemini Call",
            run_type="llm",
            inputs={"question": request.question, "history_length": len(chat.history)},
        )
        response = chat.send_message(
            request.question,
            generation_config=generation_config
        )
        prompt_tokens = response.usage_metadata.prompt_token_count
        candidates_tokens = response.usage_metadata.candidates_token_count
        total_prompt_tokens += prompt_tokens
        total_candidates_tokens += candidates_tokens
        llm_run.end(outputs=response, metadata={
            "prompt_token_count": prompt_tokens,
            "candidates_token_count": candidates_tokens,
            "total_token_count": response.usage_metadata.total_token_count
        })

        # Handle function calls
        if response.candidates and response.candidates[0].content.parts and any(p.function_call for p in response.candidates[0].content.parts):
            tool_responses = []
            # Create a span for the tool calling part
            tool_run = parent_run.create_child(
                name="Tool Calling",
                run_type="tool",
            )
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    if function_name in AVAILABLE_TOOLS:
                        function_to_call = AVAILABLE_TOOLS[function_name]
                        function_args = dict(function_call.args)
                        function_response = function_to_call(**function_args)

                        tool_responses.append(protos.Part(
                            function_response=protos.FunctionResponse(
                                name=function_name,
                                response=function_response
                            )
                        ))
            tool_run.end(outputs={"tool_responses": tool_responses})
            
            if tool_responses:
                tool_llm_run = parent_run.create_child(
                    name="Gemini Call with Tools",
                    run_type="llm",
                    inputs={"tool_responses": tool_responses},
                )
                response = chat.send_message(tool_responses)
                prompt_tokens_tool = response.usage_metadata.prompt_token_count
                candidates_tokens_tool = response.usage_metadata.candidates_token_count
                total_prompt_tokens += prompt_tokens_tool
                total_candidates_tokens += candidates_tokens_tool
                tool_llm_run.end(outputs=response, metadata={
                    "prompt_token_count": prompt_tokens_tool,
                    "candidates_token_count": candidates_tokens_tool,
                    "total_token_count": response.usage_metadata.total_token_count
                })

        final_response = response.text

        # Clean history for future calls by removing tool-related messages
        clean_history = []
        for message in chat.history:
            # Keep only parts that are text. This will filter out function_call and function_response parts.
            text_parts = [part for part in message.parts if part.text]
            if text_parts:
                # Create a new Content object with the same role but only text parts
                clean_message = protos.Content(role=message.role, parts=text_parts)
                clean_history.append(clean_message)
        
        chat.history = clean_history

        parent_run.end(outputs={"response": final_response}, metadata={
            "total_prompt_tokens": total_prompt_tokens,
            "total_candidates_tokens": total_candidates_tokens,
            "total_tokens": total_prompt_tokens + total_candidates_tokens,
        })
        return ChatResponse(response=final_response)

    except Exception as e:
        parent_run.end(error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    finally:
        # Ensure the run is always posted
        parent_run.post()

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Caseone AI Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
