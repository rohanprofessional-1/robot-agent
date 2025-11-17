"""
Basic LangChain Agent Setup
This is a simple example to get you started with LangChain.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Example: Simple calculator tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Create tools
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for performing mathematical calculations. Input should be a valid Python expression."
    )
]

async def main():
    """Main function to run the agent."""
    # Initialize the LLM (you'll need to set OPENAI_API_KEY in your environment)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Get the prompt from LangChain Hub
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Example query
    result = await agent_executor.ainvoke({
        "input": "What is 15 * 23 + 42?"
    })
    
    print("\n" + "="*50)
    print("Agent Response:")
    print("="*50)
    print(result["output"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

