"""
LangChain Agent with Robot Control
This agent can control a UR robot through natural language commands.
The agent automatically returns the robot to home position after each action.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.agents.agent_executor import AgentExecutor
from langchain.agents.create_openai_tools_agent import create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import robot controller
from robot_controller import get_robot_controller

# Initialize robot controller
ROBOT_IP = os.getenv('ROBOT_IP', '169.254.152.222')  # Can be set in .env file
robot_controller = get_robot_controller(robot_ip=ROBOT_IP)

# System prompt that enforces returning to home after actions
SYSTEM_PROMPT = """You are a helpful robot control assistant. You can control a UR robot arm through natural language commands.

IMPORTANT RULES:
1. After EVERY user action that involves moving the robot, you MUST automatically return the robot to its home position using the move_to_home tool.
2. The home position is a safe position where the robot is ready for the next command.
3. Always confirm successful movements and inform the user when returning to home.
4. If a user asks to move the robot to a position, first move to that position, then immediately return to home.
5. You can move the robot by providing 6 joint angles in degrees (one for each joint of the 6-DOF robot arm).

Available robot operations:
- move_to_position: Move robot to specified joint angles (6 values in degrees)
- move_to_home: Move robot to its home/safe position
- get_current_position: Get the current joint angles of the robot

Always be safety-conscious and confirm movements before executing them."""


def move_robot_to_position(joint_angles_str: str) -> str:
    """
    Move robot to a specified position.
    
    Args:
        joint_angles_str: Comma-separated string of 6 joint angles in degrees
                         Example: "-90, -90, 90, -90, -90, 90"
    
    Returns:
        str: Success or error message
    """
    try:
        # Parse the joint angles
        angles = [float(x.strip()) for x in joint_angles_str.split(',')]
        result = robot_controller.move_to_position(angles)
        
        # Automatically return to home after movement
        home_result = robot_controller.move_to_home()
        return f"{result}\n\nAutomatically returned to home: {home_result}"
    except ValueError as e:
        return f"Error: Invalid joint angles format. Please provide 6 comma-separated numbers. Example: '-90, -90, 90, -90, -90, 90'. Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def move_robot_to_home() -> str:
    """
    Move robot to its home position.
    
    Returns:
        str: Success or error message
    """
    return robot_controller.move_to_home()


def get_robot_position() -> str:
    """
    Get the current joint positions of the robot.
    
    Returns:
        str: Current position in degrees
    """
    return robot_controller.get_current_position()


# Create tools for the agent
tools = [
    Tool(
        name="move_robot_to_position",
        func=move_robot_to_position,
        description="Move the robot to a specific position. Input should be 6 comma-separated joint angles in degrees. Example: '-90, -90, 90, -90, -90, 90'. The robot will automatically return to home after this movement."
    ),
    Tool(
        name="move_robot_to_home",
        func=move_robot_to_home,
        description="Move the robot to its home/safe position. Use this to return the robot to a safe position."
    ),
    Tool(
        name="get_robot_position",
        func=get_robot_position,
        description="Get the current joint positions of the robot in degrees. Returns 6 values representing each joint angle."
    ),
]


async def main():
    """Main function to run the agent."""
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment.")
        print("Please set it in your .env file or environment variables.")
        return
    
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create custom prompt with system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print("=" * 70)
    print("Robot Control Agent")
    print("=" * 70)
    print("You can now control the robot using natural language commands.")
    print("Examples:")
    print("  - 'Move the robot to position -45, -45, -60, -45, -45, -45'")
    print("  - 'What is the current position of the robot?'")
    print("  - 'Move to home position'")
    print("\nThe robot will automatically return to home after each movement.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Interactive loop
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nShutting down...")
                break
            
            if not user_input:
                continue
            
            # Run the agent
            result = await agent_executor.ainvoke({
                "input": user_input
            })
            
            print("\n" + "=" * 70)
            print("Agent Response:")
            print("=" * 70)
            print(result["output"])
            print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Disconnect robot on exit
        print("\nDisconnecting robot...")
        robot_controller.disconnect()
        print("Goodbye!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
