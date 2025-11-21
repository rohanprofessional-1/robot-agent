# Robot Control Agent

This agent allows you to control a UR robot arm using natural language commands through LangChain.

## Features

- **Natural Language Control**: Control the robot using simple text commands
- **Automatic Home Return**: The robot automatically returns to home position after each movement for safety
- **Interactive Interface**: Chat with the agent to control the robot
- **Streamlit Web Interface**: Beautiful web-based chat interface for easy interaction

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Robot IP** (optional):
   - Add to your `.env` file:
     ```
     ROBOT_IP=169.254.152.222
     ```
   - Or modify the default in the code

3. **Set OpenAI API Key**:
   - Add to your `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

4. **Ensure Robot is Ready**:
   - Robot is powered on
   - Robot is connected to the network
   - Robot is in remote control mode

## Usage

### Option 1: Streamlit Web Interface (Recommended)

Run the Streamlit app:
```bash
cd agent
streamlit run app.py
```

This will open a web browser with a chat interface where you can:
- Type natural language commands
- See chat history
- View robot status
- Control settings

### Option 2: Command Line Interface

Run the agent from command line:
```bash
python agent/main.py
```

Then interact with the agent using natural language:

### Example Commands

- **Move to a position**: 
  - "Move the robot to position -45, -45, -60, -45, -45, -45"
  - "Move joints to -90, -90, 90, -90, -90, 90 degrees"

- **Pick up objects** (future functionality):
  - "Pick up the object at position -30, -30, -50, -30, -30, -30"
  - "Move to pick up the flower"
  - "Grab the item at coordinates -45, -45, -60, -45, -45, -45"

- **Get current position**:
  - "What is the current position?"
  - "Where is the robot right now?"

- **Move to home**:
  - "Move to home position"
  - "Return to home"

- **Quit** (CLI only):
  - Type `quit`, `exit`, or `q` to stop

## How It Works

1. The agent uses LangChain with OpenAI to understand your commands
2. Robot movements are handled by the `RobotController` class
3. After each movement command, the robot automatically returns to its home position
4. The home position is defined in `robot_controller.py` and can be customized

## Files

- `app.py`: Streamlit web interface for robot control (recommended)
- `main.py`: Command-line agent script with LangChain integration
- `robot_controller.py`: Robot control functions and connection management
- `move_to_pos.py`: Standalone script for direct robot movement (without agent)

## Safety Notes

- The robot will automatically return to home after each movement
- Always ensure the workspace is clear before running commands
- The robot connection is managed automatically
- Use `quit` or `exit` to properly disconnect the robot

