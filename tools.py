import sys
import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

project_root = os.path.dirname(os.path.abspath(__file__))



import search_agent.search_graph as search_graph_module
import calendar_agent.calendar_graph as calendar_graph_module
import email_agent.agent.email_graph as email_graph_module

# Load graphs using their new unique names
search_app = search_graph_module.create_search_graph()
calendar_app = calendar_graph_module.graph
email_app = email_graph_module.graph

@tool
def search_agent_tool(query: str, thread_id: str = "search_thread") -> str:
    """
    Invokes the Search Agent to search the web and retrieve current information.
    
    USE CASE:
    Use this agent whenever you need to find up-to-date information from the internet, look up current events, or answer factual questions that require web search.
    
    TOOLS IT HAS:
    - tavily_search: A search engine tool optimized for LLMs to retrieve accurate and relevant web results.
    
    WHAT IT CAN DO:
    - Perform detailed web searches to find facts and current events.
    - Synthesize information from multiple internet sources.
    - Answer questions about recent events that are not in the model's training data.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [HumanMessage(content=query)]}
    result = search_app.invoke(state, config=config)
    return result["messages"][-1].content


@tool
def calendar_agent_tool(query: str, thread_id: str = "calendar_thread") -> str:
    """
    Invokes the Calendar Agent to manage the user's Google Calendar.
    
    USE CASE:
    Use this agent whenever you need to interact with the user's schedule, meetings, or calendar events.
    
    TOOLS IT HAS:
    - calendar_tools (Google Calendar API wrapper): Provides the ability to read, create, update, and delete calendar events.
    
    WHAT IT CAN DO:
    - Check availability and list upcoming events.
    - Create new meetings or calendar events.
    - Reschedule or update existing events.
    - Delete or cancel events.
    - It understands the dynamic current time and operates in the local timezone (Asia/Kolkata).
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {"messages": [HumanMessage(content=query)]}
    result = calendar_app.invoke(state, config=config)
    return result["messages"][-1].content


@tool
def email_agent_tool(query: str, thread_id: str = "email_thread") -> str:
    """
    Invokes the Email Agent to manage the user's Gmail inbox.
    
    USE CASE:
    Use this agent for reading, searching, summarizing, drafting, and managing emails.
    
    TOOLS IT HAS:
    - list_emails: Fetch recent emails or search the inbox using a query.
    - summarize_email: Generate summaries of specific long emails.
    - filter_email: Apply specific filters to search for emails matching certain criteria.
    - create_email: Draft new emails to specified recipients.
    - reply_email: Draft replies to existing email threads.
    
    WHAT IT CAN DO:
    - Check the inbox for new or specific messages.
    - Summarize long threads or important emails.
    - Draft new emails or replies and save them as drafts in Gmail.
    - Features a human-in-the-loop review process where the draft is presented to the user for feedback before saving.
    """
    from email_agent.email_connector import fetch_emails
    recent_emails = fetch_emails()
    
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "emails": recent_emails,
        "messages": [HumanMessage(content=query)],
        "revision_count": 0
    }
    result = email_app.invoke(state, config=config)
    return result["messages"][-1].content

# Export a list of all sub-agent tools so the supervisor can bind them
supervisor_tools = [search_agent_tool, calendar_agent_tool, email_agent_tool]