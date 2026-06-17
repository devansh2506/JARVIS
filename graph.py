import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import the tools we wrapped in tools.py
from tools import supervisor_tools
from prompt import SUPERVISOR_PROMPT

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.0
)

# Bind the supervisor tools to the LLM
model_with_tools = llm.bind_tools(supervisor_tools)

def call_model(state: MessagesState):
    """
    Calls the supervisor model.
    """
    messages = state["messages"]
    
    # Prepend the system prompt dynamically
    messages_for_llm = [SystemMessage(content=SUPERVISOR_PROMPT)] + messages
    
    response = model_with_tools.invoke(messages_for_llm)
    
    return {"messages": [response]}

def should_continue(state: MessagesState):
    """
    Conditional edge: routes to tools if a tool call is made, otherwise routes to END.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, we go to the tool node
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
        
    # If no tool call, we end and return the final response to the user
    return END

# Set up the ToolNode with our supervisor tools
tool_node = ToolNode(supervisor_tools)

# Define the StateGraph using MessagesState (which tracks conversation history)
workflow = StateGraph(MessagesState)

# Add our two nodes: the LLM (agent) and the tools
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entry point to always start at the agent
workflow.add_edge(START, "agent")

# Add the conditional edges from the agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Once the tools finish running, always return to the agent to process the tool outputs
workflow.add_edge("tools", "agent")

# Set up a checkpointer to persist conversation memory
memory = MemorySaver()

# Compile the graph
graph = workflow.compile(checkpointer=memory)
