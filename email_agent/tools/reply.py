import os
import sys
from typing import Annotated

# Add the 'email_agent' root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

REPLY_EMAIL_PROMPT = """<>
You are an expert email copywriter. You need to write a professional reply to the following email.

Original Email Sender: {sender}
Original Email Subject: {subject}
Original Email Content: {original_content}

Instructions/Content for the reply (if any):
{content}

If no instructions are provided above, generate an appropriate, polite, and professional response acknowledging the email.

Drafted Reply:
</>
"""

@tool
def reply_email(email_id: str, content: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Generates a professional reply to an existing email.
    
    Args:
        email_id (str): The ID of the email to reply to.
        content (str): Optional instructions or specific content to include in the reply. If not provided, pass an empty string.
        state (dict): The injected graph state containing fetched emails.
        
    Returns:
        str: The full text of the drafted reply email.
    """
    emails = state.get("emails", [])
    email = next((e for e in emails if e.get("id") == email_id), None)
    
    if not email:
        return f"Error: Email with ID {email_id} not found."
        
    prompt_template = PromptTemplate(
        input_variables=["sender", "subject", "original_content", "content"],
        template=REPLY_EMAIL_PROMPT
    )
    
    formatted_prompt = prompt_template.format(
        sender=email.get("from", "Unknown Sender"),
        subject=email.get("subject", "No Subject"),
        original_content=email.get("body", "No Content"),
        content=content if content else "No specific instructions provided. Please draft a polite, standard reply."
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.0
    )
    
    response = llm.invoke(formatted_prompt) 
    
    return response.content
