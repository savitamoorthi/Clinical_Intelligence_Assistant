"""
Clinical Intelligence Assistant API.

This module defines the FastAPI application, configures CORS, 
and exposes the primary chat and health endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from agent.supervisor import app
from langchain_core.messages import HumanMessage
import uuid

# FastAPI setup
api = FastAPI(title="Clinical Intelligence Assistant")

# CORS middleware 
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    role: str
    content: str
    session_id: str
    metadata: Dict[str, Any] = {}


# Health check endpoint
@api.get("/health")
def health_check():
    return {"status": "ok", "service": "Clinical Intelligence Assistant"}

# Main chat endpoint 
@api.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        user_input = {"messages": [HumanMessage(content=request.message)]}
        final_state = app.invoke(user_input, config=config)
        ai_message = final_state['messages'][-1].content
    
        metadata = {
            "sql_data_present": bool(final_state.get("sql_data")),
            "report_data_present": bool(final_state.get("report_data")),
            "patient_ids": final_state.get("target_subject_ids", [])
        }

        return ChatResponse(
            role="assistant",
            content=ai_message,
            session_id=session_id,
            metadata=metadata
        )

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)