"""
Streamlit Chat Interface for Robot Control Agent with Vision Tools

Control a UR robot arm and a vision pipeline (camera + test tube detection)
through natural language. Includes simple conversational memory via
st.session_state.messages.
"""

import os
import sys
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

# ---------------------------------------------------------------------
# Environment & imports
# ---------------------------------------------------------------------

load_dotenv()

# Add parent directory to path so imports work when run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_controller import get_robot_controller
from image_processor import get_image_processor


# ---------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful robot control assistant. You can control a UR robot arm through natural language commands. Additionally you are equipped with a camera and a vision pipeline to process pictures located in the directory 'logs'.

IMPORTANT RULES:
1. After EVERY user action that involves moving the robot, you MUST automatically return the robot to its home position using the move_to_home tool.
2. The home position is a safe position where the robot is ready for the next command.
3. Always confirm successful movements and inform the user when returning to home.
4. If a user asks to move the robot to a position, first move to that position, then immediately return to home.
5. You can move the robot by providing 6 joint angles in degrees (one for each joint of the 6-DOF robot arm).

Available robot operations:
- move_robot_to_position: Move robot to specified joint angles (6 values in degrees).
- move_robot_to_home: Move robot to its home/safe position.
- get_robot_position: Get the current joint angles of the robot.

Available vision operations:
- capture_image: Capture an image using the robot's camera. Use this ONLY when the user specifically asks to take a picture or no recent image is available. Returns the filename of the saved image.
- detect_test_tubes: Detect all test tubes in an image. If no image_path is provided, uses the latest captured image.
- find_tubes_by_color: Find all test tubes of a given color in the latest image. Returns a list of dicts with tube_id and pixel/world coordinates of the tube center.

Always be safety-conscious and confirm movements before executing them.
Only call vision tools when needed, and avoid taking unnecessary new images.
"""


# ---------------------------------------------------------------------
# Agent / tools initialization
# ---------------------------------------------------------------------

def initialize_agent():
    """Initialize LLM with bound robot + vision tools, and controllers."""
    robot_ip = os.getenv("ROBOT_IP", "169.254.152.222")
    robot_controller = get_robot_controller(robot_ip=robot_ip)

    model_api = os.getenv("ROBOFLOW_API_KEY")
    image_processor = get_image_processor(model_api=model_api)

    # ---------------- Robot tools ----------------

    def move_robot_to_position(joint_angles_str: str) -> str:
        """Move robot to 6 comma-separated joint angles in degrees, then home."""
        try:
            angles = [float(x.strip()) for x in joint_angles_str.split(",")]
            if len(angles) != 6:
                raise ValueError("Exactly 6 joint angles are required.")
            result = robot_controller.move_to_position(angles)
            home_result = robot_controller.move_to_home()
            return f"{result}\n\n✅ Automatically returned to home: {home_result}"
        except ValueError as e:
            return (
                "❌ Error: Invalid joint angles format. Please provide 6 comma-separated "
                "numbers (e.g. '-90, -90, 90, -90, -90, 90'). "
                f"Details: {e}"
            )
        except Exception as e:
            return f"❌ Error: {e}"

    def move_robot_to_home(*args, **kwargs) -> str:
        """Return robot to its home/safe position."""
        return robot_controller.move_to_home()

    def get_robot_position(*args, **kwargs) -> str:
        """Get current joint positions of the robot in degrees."""
        return robot_controller.get_current_position()

    # ---------------- Vision tools ----------------

    def capture_image(*args, **kwargs) -> str:
        """
        Capture a new workspace image from the robot camera.

        Returns:
            str: Path to the saved image. Also updates image_processor.last_image_path.
        """
        return image_processor.capture_image()

    def detect_test_tubes(image_path: str | None = None) -> list[dict]:
        """
        Detect all test tubes in an image.

        Args:
            image_path: Optional path to an image file. If None or empty,
                        uses the latest captured image, or captures a new
                        one if none exists.

        Returns:
            list of dicts with:
                tube_id, bbox (x1,y1,x2,y2), pixel_center (cx,cy),
                world_coords (X,Y,Z), conf.
        """
        if not image_path:
            image_path = getattr(image_processor, "last_image_path", None)
            if image_path is None:
                image_path = image_processor.capture_image()

        tubes = image_processor.detect_test_tubes(image_path=image_path)
        frame_bgr = image_processor.load_image(image_path)

        result = []
        for tube in tubes:
            x1, y1, x2, y2 = tube.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            X, Y, Z = image_processor.image_to_coords(cx, cy)
            result.append(
                {
                    "tube_id": tube.tube_id,
                    "bbox": (x1, y1, x2, y2),
                    "pixel_center": (cx, cy),
                    "world_coords": (X, Y, Z),
                    "conf": tube.conf,
                }
            )

        # Cache last detections for potential reuse
        image_processor.last_image_path = image_path
        image_processor.last_detections = result
        return result

    def find_tubes_by_color(color_name: str) -> list[dict]:
        """
        Find all test tubes of a given color in the latest image.

        Args:
            color_name: Color name string like 'red', 'blue', 'green', etc.

        Returns:
            list of dicts with:
                tube_id, color, pixel_center (cx,cy),
                world_coords (X,Y,Z), conf.
        """
        image_path = getattr(image_processor, "last_image_path", None)
        if image_path is None:
            image_path = image_processor.capture_image()

        tubes = image_processor.get_colored_tubes(image_path, color_name)
        result = []
        for tube in tubes:
            x1, y1, x2, y2 = tube.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            X, Y, Z = image_processor.image_to_coords(cx, cy)
            result.append(
                {
                    "tube_id": tube.tube_id,
                    "color": tube.color,
                    "pixel_center": (cx, cy),
                    "world_coords": (X, Y, Z),
                    "conf": tube.conf,
                }
            )

        # Keep same last_image_path
        image_processor.last_image_path = image_path
        image_processor.last_detections = result
        return result

    # ---------------- Tool registration ----------------

    tools = [
        Tool(
            name="move_robot_to_position",
            func=move_robot_to_position,
            description=(
                "Move the robot to a specific position. "
                "Input: a string with 6 comma-separated joint angles in degrees "
                "(e.g. '-90, -90, 90, -90, -90, 90'). "
                "Robot automatically returns to home afterward."
            ),
        ),
        Tool(
            name="move_robot_to_home",
            func=move_robot_to_home,
            description="Move the robot to its home/safe position.",
        ),
        Tool(
            name="get_robot_position",
            func=get_robot_position,
            description="Get the current joint positions of the robot in degrees.",
        ),
        Tool(
            name="capture_image",
            func=capture_image,
            description=(
                "Capture a new image of the workspace using the robot camera. "
                "Returns the path to the saved image. "
                "Use only when the user explicitly requests a new picture, "
                "or when no recent image is available."
            ),
        ),
        Tool(
            name="detect_test_tubes",
            func=detect_test_tubes,
            description=(
                "Detect all test tubes in an image. "
                "Input: optional image_path string. If omitted, uses the latest captured image. "
                "Returns a list of tubes with tube_id, bbox, pixel_center, world_coords, and conf."
            ),
        ),
        Tool(
            name="find_tubes_by_color",
            func=find_tubes_by_color,
            description=(
                "Find all test tubes of a given color in the latest image. "
                "Input: a color name string like 'red', 'blue', 'green', or 'yellow'. "
                "Returns a list of tubes with tube_id, color, pixel_center, world_coords, and conf."
            ),
        ),
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}

    return llm_with_tools, tool_map, robot_controller


# ---------------------------------------------------------------------
# Robot state tracking (for sidebar)
# ---------------------------------------------------------------------

def update_robot_state(position: str):
    """Update robot task state in session."""
    if "robot_state" not in st.session_state:
        st.session_state.robot_state = {
            "last_position": None,
            "completed_tasks": [],
            "active_goal": None,
        }
    st.session_state.robot_state["last_position"] = position
    st.session_state.robot_state["completed_tasks"].append(
        {
            "timestamp": str(datetime.now()),
            "action": "move",
            "position": position,
        }
    )


# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Robot Control Agent",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Robot Control Agent with Vision Tools")
    st.markdown(
        "Control your robot arm and vision pipeline using natural language commands. "
        "**The agent remembers the conversation within this session.**"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.agent_initialized = False
        st.session_state.robot_state = {
            "last_position": None,
            "completed_tasks": [],
            "active_goal": None,
        }

    # Initialize / refresh agent and tools each run
    try:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("⚠️ OPENAI_API_KEY not found. Please set it in your .env file.")
            st.stop()

        llm_with_tools, tool_map, robot_controller = initialize_agent()
        st.session_state.llm = llm_with_tools
        st.session_state.tool_map = tool_map
        st.session_state.robot_controller = robot_controller

        if not st.session_state.agent_initialized:
            st.session_state.agent_initialized = True
            st.success("✅ Agent initialized successfully!")

    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(
        "Type your command here (e.g., 'Move the robot to position -45, -45, -60, -45, -45, -45')"
    ):
        # Log user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing..."):
                try:
                    messages = [SystemMessage(content=SYSTEM_PROMPT)]
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(AIMessage(content=msg["content"]))

                    # First LLM call (may include tool calls)
                    response = st.session_state.llm.invoke(messages)

                    # Tool loop
                    while getattr(response, "tool_calls", None):
                        messages.append(response)
                        for tool_call in response.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            if tool_name in st.session_state.tool_map:
                                tool = st.session_state.tool_map[tool_name]
                                tool_result = tool.invoke(tool_args)
                                messages.append(
                                    ToolMessage(
                                        tool_call_id=tool_call["id"],
                                        content=str(tool_result),
                                    )
                                )
                            else:
                                messages.append(
                                    ToolMessage(
                                        tool_call_id=tool_call["id"],
                                        content="Error: Tool not found",
                                    )
                                )

                        # Call LLM again with tool results
                        response = st.session_state.llm.invoke(messages)

                    response_text = response.content
                    st.markdown(response_text)

                    # Save assistant response
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                    # Update robot state
                    current_pos = st.session_state.robot_controller.get_current_position()
                    update_robot_state(current_pos)

                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.markdown(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown(
            """
        **How to use:**
        1. Type your command in natural language.
        2. The agent can move the robot and use the camera/vision tools.
        3. The robot automatically returns to home after each movement.

        **Example commands:**
        - "Move the robot to position -45, -45, -60, -45, -45, -45"
        - "What is the current position?"
        - "Take a picture of the workspace"
        - "Detect all test tubes in the latest image"
        - "Find all blue test tubes and tell me their coordinates"
        """
        )

        st.header("🧠 State")
        if st.button("📊 Show Robot State"):
            st.json(st.session_state.robot_state)

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.header("🔒 Safety")
        st.info(
            "The robot will automatically return to the home position after each movement for safety."
        )

        if st.button("Disconnect Robot"):
            if st.session_state.robot_controller:
                st.session_state.robot_controller.disconnect()
                st.success("Robot disconnected")
            else:
                st.warning("Robot not connected")


if __name__ == "__main__":
    main()
