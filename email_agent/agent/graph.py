import os
import sys

# Ensure the parent directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
import json
from pydantic import BaseModel, Field

class EmailDraft(BaseModel):
    receiver_id: str = Field(description="The email address of the receiver")
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The main content of the email")

from agent.state import AgentState
from agent.prompt import SYSTEM_PROMPT
from tools.summariser import summarize_email
from tools.create import create_email
from tools.filtering import filter_email
from tools.list_emails import list_emails
from tools.reply import reply_email
from email_connector import save_as_draft, get_gmail_service

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gather tools
tools = [summarize_email, filter_email, create_email, list_emails, reply_email]

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.0
)

# Bind tools to the model
model = llm.bind_tools(tools)

def call_model(state: AgentState):
    """
    Calls the model with the current messages and the system prompt.
    """
    messages = state["messages"]
    
    # Prepend the system prompt
    messages_to_invoke = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = model.invoke(messages_to_invoke)
    
    return {"messages": [response]}

def review_node(state: AgentState):
    """
    Shows the draft from the tool and asks for changes.
    """
    messages = state["messages"]
    # The last message is the ToolMessage with the draft
    draft_msg = messages[-1]
    
    print("\n--- EMAIL DRAFT PENDING REVIEW ---")
    print(f"{draft_msg.content}")
    print("----------------------------------")
    
    feedback = input("\nProvide feedback for changes (or hit enter to approve): ")
    if not feedback.strip():
        feedback = "Looks good, no revisions needed. Please finalize the email as is."
    else:
        feedback = f"Please revise the draft based on this feedback: {feedback}"
        
    return {"messages": [HumanMessage(content=feedback)]}

def format_and_save_node(state: AgentState):
    """
    Uses structured output to parse the draft + feedback into a final JSON.
    Then saves it as a draft in Gmail.
    """
    messages = state["messages"]
    
    structured_llm = llm.with_structured_output(EmailDraft)
    print("\nJARVIS is finalizing the draft...")
    
    system_msg = SystemMessage(content="You are an expert email assistant. Extract the final email details based on the conversation history. Apply any requested revisions from the user's last message.")
    messages_to_invoke = [system_msg] + messages
    
    try:
        final_draft: EmailDraft = structured_llm.invoke(messages_to_invoke)
        
        print("\nSaving draft to Gmail...")
        service = get_gmail_service()
        draft_result = save_as_draft(service, final_draft.receiver_id, final_draft.subject, final_draft.body)
        
        if draft_result:
            success_msg = AIMessage(content=f"Email drafted successfully! Draft ID: {draft_result['id']}. I have saved it to your Gmail.")
        else:
            success_msg = AIMessage(content="Failed to save the draft. Check console for details.")
            
    except Exception as e:
        success_msg = AIMessage(content=f"An error occurred while formatting or saving the draft: {str(e)}")
        
    return {"messages": [success_msg]}

def should_continue(state: AgentState):
    """
    Conditional edge routing:
    If the model calls a tool, route to 'tools'.
    If not, route to END.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
            
    return "end"

def after_tools(state: AgentState):
    """
    Conditional edge routing from tools:
    If create_email or reply_email was used, route directly to review_node.
    Otherwise, go back to agent.
    """
    messages = state["messages"]
    
    # Check all ToolMessages that were just executed
    tool_names = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            break  # We reached the AIMessage that triggered the tools
        if isinstance(msg, ToolMessage):
            tool_names.append(msg.name)
    
    if "create_email" in tool_names or "reply_email" in tool_names:
        return "review"
        
    return "agent"

# Set up the ToolNode
tool_node = ToolNode(tools)

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("review_node", review_node)
workflow.add_node("format_and_save_node", format_and_save_node)

# Set the entry point
workflow.add_edge(START, "agent")

# Add conditional edges from 'agent'
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    }
)

# Add conditional edges from 'tools'
workflow.add_conditional_edges(
    "tools",
    after_tools,
    {
        "review": "review_node",
        "agent": "agent",
    }
)

# Flow from review to format_and_save, then to end
workflow.add_edge("review_node", "format_and_save_node")
workflow.add_edge("format_and_save_node", END)

# Set up Checkpointer
memory = MemorySaver()

# Compile the graph
graph = workflow.compile(
    checkpointer=memory
)