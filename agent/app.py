"""
Streamlit Chat Interface for Robot Control Agent with Persistent Memory
A simple chat interface for controlling the robot arm through natural language with conversation memory.
"""

import streamlit as st
import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents.openai_tools.base import create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver  # 🔄 NEW: LangGraph memory
from langgraph.prebuilt import create_react_agent  # 🔄 NEW: LangGraph agent
from langchain_community.chat_message_histories import StreamlitChatMessageHistory  # 🔄 NEW: Streamlit history
from langchain_core.messages import HumanMessage  # 🔄 NEW: For message handling

# Import robot controller
from robot_controller import get_robot_controller
# Import image processor
from image_processor import get_image_processor

# System prompt (unchanged)
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
    """Initialize the robot control agent with persistent memory."""  # 🔄 CHANGED: Added memory support
    robot_ip = os.getenv('ROBOT_IP', '169.254.152.222')
    
    robot_controller = get_robot_controller(robot_ip=robot_ip)
    model_api = os.getenv('ROBOFLOW_API_KEY')
    image_processor = get_image_processor(model_api=model_api)
    # Robot control functions (MINOR: Better error formatting)
    def move_robot_to_position(joint_angles_str: str) -> str:
        try:
            angles = [float(x.strip()) for x in joint_angles_str.split(',')]
            result = robot_controller.move_to_position(angles)
            home_result = robot_controller.move_to_home()
            return f"{result}\n\n✅ Automatically returned to home: {home_result}"
        except ValueError as e:
            return f"❌ Error: Invalid joint angles format. Please provide 6 comma-separated numbers. Example: '-90, -90, 90, -90, -90, 90'. Error: {str(e)}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def move_robot_to_home() -> str:
        return robot_controller.move_to_home()
    
    def get_robot_position() -> str:
        return robot_controller.get_current_position()
    
    # Create tools (MINOR: Better formatting)
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
    
    # Initialize persistent message history  # 🔄 NEW
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = StreamlitChatMessageHistory(key="robot_chat")
    
    # Create persistent checkpointer  # 🔄 NEW
    checkpointer = MemorySaver()
    
    # Create agent with memory  # 🔄 CHANGED: LangGraph instead of OpenAI tools agent
    agent = create_react_agent(llm, tools, state_modifier=SYSTEM_PROMPT)
    
    # Config with thread_id for session persistence  # 🔄 NEW
    config = {"configurable": {"thread_id": "robot_session"}}
    
    return agent, checkpointer, config, robot_controller, st.session_state.chat_history

def update_robot_state(position: str):  # 🔄 NEW: Robot task state tracking
    """Update robot task state in session."""
    if "robot_state" not in st.session_state:
        st.session_state.robot_state = {
            "last_position": None,
            "completed_tasks": [],
            "active_goal": None
        }
    st.session_state.robot_state["last_position"] = position
    st.session_state.robot_state["completed_tasks"].append({
        "timestamp": str(datetime.now()),
        "action": "move",
        "position": position
    })

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Robot Control Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Robot Control Agent with Memory")
    st.markdown("Control your robot arm using natural language commands. **The agent now remembers previous conversations!**")  # 🔄 CHANGED: Added memory mention
    
    # Initialize session state  # 🔄 CHANGED: Added robot_state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.agent_initialized = False
        st.session_state.robot_state = {  # 🔄 NEW
            "last_position": None,
            "completed_tasks": [],
            "active_goal": None
        }
    
    # Initialize agent on first run  # 🔄 CHANGED: Memory-enabled initialization
    if not st.session_state.agent_initialized:
        with st.spinner("🤖 Initializing robot agent with persistent memory..."):
            try:
                # Check for API key
                if not os.getenv("OPENAI_API_KEY"):
                    st.error("⚠️ OPENAI_API_KEY not found. Please set it in your .env file.")
                    st.stop()
                
                agent_executor, checkpointer, config, robot_controller, chat_history = initialize_agent()
                
                # Store in session state  # 🔄 CHANGED: Store all memory components
                st.session_state.agent_executor = agent_executor
                st.session_state.checkpointer = checkpointer
                st.session_state.config = config
                st.session_state.robot_controller = robot_controller
                st.session_state.chat_history = chat_history
                st.session_state.agent_initialized = True
                st.success("✅ Agent with **persistent memory** initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize agent: {str(e)}")
                st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input  # 🔄 CHANGED: Memory-enabled agent invocation
    if prompt := st.chat_input("Type your command here (e.g., 'Move the robot to position -45, -45, -60, -45, -45, -45')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response WITH MEMORY  # 🔄 COMPLETELY NEW LOGIC
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing command with memory..."):
                try:
                    agent, checkpointer, config, robot_controller = (
                        st.session_state.agent_executor,
                        st.session_state.checkpointer,
                        st.session_state.config,
                        st.session_state.robot_controller
                    )
                    
                    # Run agent with persistent memory  # 🔄 NEW: LangGraph invoke pattern
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config=config,
                        checkpointer=checkpointer
                    )
                    
                    # Extract final response  # 🔄 NEW: Extract from messages
                    response = result["messages"][-1].content
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Update robot state if position changed  # 🔄 NEW
                    current_pos = robot_controller.get_current_position()
                    update_robot_state(current_pos)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with information  # 🔄 CHANGED: Added memory controls
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        **How to use:**
        1. Type your command in natural language
        2. The agent will **remember** previous conversations
        3. Robot automatically returns to home after each movement
        
        **Example multi-turn conversation:**
        - "Move the robot to position -45, -45, -60, -45, -45, -45"
        - "Now move it 10 degrees higher on joint 3"
        - "What is the current position?"
        """)
        
        st.header("⚙️ Settings")
        robot_ip = st.text_input(
            "Robot IP Address",
            value=os.getenv('ROBOT_IP', '169.254.152.222'),
            help="IP address of your UR robot"
        )
        
        # 🔄 NEW: Memory management buttons
        st.header("🧠 Memory Controls")
        if st.button("🧹 Clear Agent Memory"):
            st.session_state.chat_history.clear()
            st.success("✅ Conversation memory cleared!")
            st.rerun()
        st.caption("Clears agent memory but keeps robot connection")
        
        if st.button("📊 Show Robot State"):  # 🔄 NEW
            st.json(st.session_state.robot_state)
        
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