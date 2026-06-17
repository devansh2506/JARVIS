import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CREATE_EMAIL_PROMPT = """<>
You are an expert email copywriter. Given the brief provided below, generate a professional, clear, and engaging email.

Brief: {brief}

Email Content:
</>
"""

@tool
def create_email(brief: str) -> str:
    """
    Generates a complete, professional email draft based on a short description or brief.
    Use this tool when you need to write a brand new email or reply to an existing one.

    Args:
        brief (str): Instructions or a short description of what the email should say.

    Returns:
        str: The full text of the drafted professional email.
    """
    prompt_template = PromptTemplate(
        input_variables=["brief"],
        template=CREATE_EMAIL_PROMPT
    )
    
    formatted_prompt = prompt_template.format(brief=brief)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.7
    )
    
    response = llm.invoke(formatted_prompt) 
    
    return response.content

if __name__ == "__main__":
    pass
