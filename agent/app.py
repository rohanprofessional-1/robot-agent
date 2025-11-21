"""
Streamlit Chat Interface for Robot Control Agent
A simple chat interface for controlling the robot arm through natural language.
"""

import streamlit as st
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents.agent_executor import AgentExecutor
from langchain.agents.create_openai_tools_agent import create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import robot controller
from robot_controller import get_robot_controller

# System prompt
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


def initialize_agent():
    """Initialize the robot control agent."""
    # Initialize robot controller
    robot_ip = os.getenv('ROBOT_IP', '169.254.152.222')
    robot_controller = get_robot_controller(robot_ip=robot_ip)
    
    # Robot control functions
    def move_robot_to_position(joint_angles_str: str) -> str:
        try:
            angles = [float(x.strip()) for x in joint_angles_str.split(',')]
            result = robot_controller.move_to_position(angles)
            home_result = robot_controller.move_to_home()
            return f"{result}\n\nAutomatically returned to home: {home_result}"
        except ValueError as e:
            return f"Error: Invalid joint angles format. Please provide 6 comma-separated numbers. Example: '-90, -90, 90, -90, -90, 90'. Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def move_robot_to_home() -> str:
        return robot_controller.move_to_home()
    
    def get_robot_position() -> str:
        return robot_controller.get_current_position()
    
    # Create tools
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
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor, robot_controller


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Robot Control Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Robot Control Agent")
    st.markdown("Control your robot arm using natural language commands. The robot will automatically return to home after each movement.")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.agent_initialized = False
        st.session_state.agent_executor = None
        st.session_state.robot_controller = None
    
    # Initialize agent on first run
    if not st.session_state.agent_initialized:
        with st.spinner("Initializing robot control agent..."):
            try:
                # Check for API key
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("⚠️ OPENAI_API_KEY not found. Please set it in your .env file.")
                    st.stop()
                
                agent_executor, robot_controller = initialize_agent()
                st.session_state.agent_executor = agent_executor
                st.session_state.robot_controller = robot_controller
                st.session_state.agent_initialized = True
                st.success("✅ Agent initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize agent: {str(e)}")
                st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your command here (e.g., 'Move the robot to position -45, -45, -60, -45, -45, -45')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    # Run agent (handle async properly for Streamlit)
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        st.session_state.agent_executor.ainvoke({
                            "input": prompt
                        })
                    )
                    response = result["output"]
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        **How to use:**
        1. Type your command in natural language
        2. The agent will understand and execute robot movements
        3. Robot automatically returns to home after each movement
        
        **Example commands:**
        - "Move the robot to position -45, -45, -60, -45, -45, -45"
        - "What is the current position?"
        - "Move to home position"
        - "Pick up the object at position -30, -30, -50, -30, -30, -30"
        """)
        
        st.header("⚙️ Settings")
        robot_ip = st.text_input(
            "Robot IP Address",
            value=os.getenv('ROBOT_IP', '169.254.152.222'),
            help="IP address of your UR robot"
        )
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.header("🔒 Safety")
        st.info("The robot will automatically return to home position after each movement for safety.")
        
        if st.button("Disconnect Robot"):
            if st.session_state.robot_controller:
                st.session_state.robot_controller.disconnect()
                st.success("Robot disconnected")
            else:
                st.warning("Robot not connected")


if __name__ == "__main__":
    main()

