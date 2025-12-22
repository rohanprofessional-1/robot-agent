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

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# Import robot controller
from robot_controller import get_robot_controller
# Import image processor
from image_processor import get_image_processor

# System prompt (unchanged)
SYSTEM_PROMPT = """You are a helpful robot control assistant. You can control a UR robot arm through natural language commands. Additionally you are equipped with a camera and a vision pipeline to process pictures located in the directory logs.

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
- capture_image: Capture an image using the robot's camera

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
    
    def move_robot_to_home(*args, **kwargs) -> str:
        return robot_controller.move_to_home()
    
    def get_robot_position(*args, **kwargs) -> str:
        return robot_controller.get_current_position()
    
    def capture_image(*args, **kwargs) -> str:
        return image_processor.capture_image()




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
        Tool(
            name="capture_image",
            func=capture_image,
            description="Capture an image using the robot's camera. Returns the filename of the saved image."
        )
    ]
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}
    
    return llm_with_tools, tool_map, robot_controller

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
    
    # Initialize/Refresh agent logic on every run to ensure code changes are applied
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OPENAI_API_KEY not found. Please set it in your .env file.")
            st.stop()
        
        llm_with_tools, tool_map, robot_controller = initialize_agent()
        
        # Update session state with fresh tools and controller
        st.session_state.llm = llm_with_tools
        st.session_state.tool_map = tool_map
        st.session_state.robot_controller = robot_controller
        
        if not st.session_state.agent_initialized:
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
                    # Construct message history
                    messages = [SystemMessage(content=SYSTEM_PROMPT)]
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(AIMessage(content=msg["content"]))
                    
                    # Run agent loop
                    response = st.session_state.llm.invoke(messages)
                    
                    while response.tool_calls:
                        messages.append(response)
                        for tool_call in response.tool_calls:
                            tool_name = tool_call["name"]
                            if tool_name in st.session_state.tool_map:
                                tool_result = st.session_state.tool_map[tool_name].invoke(tool_call["args"])
                                messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_result)))
                            else:
                                messages.append(ToolMessage(tool_call_id=tool_call["id"], content="Error: Tool not found"))
                        
                        response = st.session_state.llm.invoke(messages)
                    
                    response_text = response.content
                    st.markdown(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    # Update robot state if position changed  # 🔄 NEW
                    current_pos = st.session_state.robot_controller.get_current_position()
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