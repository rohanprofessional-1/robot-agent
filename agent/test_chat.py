"""
Simple LangChain Example - Synchronous Version
A basic example that doesn't require async/await.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def simple_chat_example():
    """Simple chat example with LangChain."""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment.")
        print("Please set it in your .env file or environment variables.")
        return
    
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create a simple message
    messages = [HumanMessage(content="What is LangChain in one sentence?")]
    
    # Get response
    response = llm.invoke(messages)
    
    print("\n" + "="*50)
    print("LangChain Response:")
    print("="*50)
    print(response.content)
    print("="*50 + "\n")

if __name__ == "__main__":
    simple_chat_example()

