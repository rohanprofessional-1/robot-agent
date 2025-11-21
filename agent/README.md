# Robot Control Agent

This agent allows you to control a UR robot arm using natural language commands through LangChain.

## Features

- **Natural Language Control**: Control the robot using simple text commands
- **Automatic Home Return**: The robot automatically returns to home position after each movement for safety
- **Interactive Interface**: Chat with the agent to control the robot

## Setup

1. **Set Robot IP** (optional):
   - Add to your `.env` file:
     ```
     ROBOT_IP=169.254.152.222
     ```
   - Or modify the default in `main.py`

2. **Set OpenAI API Key**:
   - Add to your `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

3. **Ensure Robot is Ready**:
   - Robot is powered on
   - Robot is connected to the network
   - Robot is in remote control mode

## Usage

Run the agent:
```bash
python agent/main.py
```

Then interact with the agent using natural language:

### Example Commands

- **Move to a position**: 
  - "Move the robot to position -45, -45, -60, -45, -45, -45"
  - "Move joints to -90, -90, 90, -90, -90, 90 degrees"

- **Get current position**:
  - "What is the current position?"
  - "Where is the robot right now?"

- **Move to home**:
  - "Move to home position"
  - "Return to home"

- **Quit**:
  - Type `quit`, `exit`, or `q` to stop

## How It Works

1. The agent uses LangChain with OpenAI to understand your commands
2. Robot movements are handled by the `RobotController` class
3. After each movement command, the robot automatically returns to its home position
4. The home position is defined in `robot_controller.py` and can be customized

## Files

- `main.py`: Main agent script with LangChain integration
- `robot_controller.py`: Robot control functions and connection management
- `move_to_pos.py`: Standalone script for direct robot movement (without agent)

## Safety Notes

- The robot will automatically return to home after each movement
- Always ensure the workspace is clear before running commands
- The robot connection is managed automatically
- Use `quit` or `exit` to properly disconnect the robot

