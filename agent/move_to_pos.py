"""
Simple URBasic Script to Move Robot to Position
This script demonstrates how to connect to a UR robot and move it to a specified position.
"""

import sys
import os
import math
import time

# Add parent directory to path to import URBasic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import URBasic.robotModel
import URBasic.urScriptExt

# Robot Configuration
ROBOT_IP = '169.254.152.222'  # Change this to your robot's IP address
ACCEL = 0.5  # Acceleration (rad/s^2)
VEL = 0.6   # Velocity (rad/s)

# Example positions (in radians)
# These are example joint angles - adjust based on your needs
START_POS = (
    math.radians(-90),
    math.radians(-90),
    math.radians(90),
    math.radians(-90),
    math.radians(-90),
    math.radians(90)
)

# Example target position
TARGET_POS = (
    math.radians(-45),
    math.radians(-45),
    math.radians(-60),
    math.radians(-45),
    math.radians(-45),
    math.radians(-45)
)


def move_robot_to_position(robot, target_position, accel=ACCEL, vel=VEL):
    """
    Move robot to a target joint position.
    
    Args:
        robot: URBasic robot instance
        target_position: Tuple of 6 joint angles in radians
        accel: Acceleration (rad/s^2)
        vel: Velocity (rad/s)
    """
    try:
        print(f"Moving robot to position: {[math.degrees(angle) for angle in target_position]}")
        robot.movej(q=target_position, a=accel, v=vel)
        print("Movement completed successfully!")
        time.sleep(1)  # Wait for movement to complete
    except Exception as e:
        print(f"Error during movement: {e}")
        raise


def get_current_position(robot):
    """
    Get the current joint positions of the robot.
    
    Args:
        robot: URBasic robot instance
    
    Returns:
        Tuple of 6 joint angles in radians
    """
    try:
        joints = robot.get_actual_joint_positions()
        print(f"Current joint positions (degrees): {[math.degrees(j) for j in joints]}")
        return joints
    except Exception as e:
        print(f"Error getting current position: {e}")
        return None


def main():
    """Main function to initialize robot and move to position."""
    print("=" * 50)
    print("URBasic Robot Movement Script")
    print("=" * 50)
    
    # Initialize robot model
    print("\nInitializing robot model...")
    robotModel = URBasic.robotModel.RobotModel()
    
    # Initialize robot connection
    print(f"Connecting to robot at {ROBOT_IP}...")
    try:
        robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)
        print("Robot connected successfully!")
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        print("Please check:")
        print("  1. Robot IP address is correct")
        print("  2. Robot is powered on and connected to network")
        print("  3. Robot is in remote control mode")
        return
    
    try:
        # Reset any errors
        print("\nResetting robot errors...")
        robot.reset_error()
        
        # Initialize real-time control
        print("Initializing real-time control...")
        robot.init_realtime_control()
        time.sleep(1)
        
        # Get current position
        print("\n" + "-" * 50)
        print("Current Position:")
        current_pos = get_current_position(robot)
        
        # Move to start position (optional)
        print("\n" + "-" * 50)
        print("Moving to start position...")
        move_robot_to_position(robot, START_POS)
        
        # Get position after start move
        print("\n" + "-" * 50)
        print("Position after start move:")
        get_current_position(robot)
        
        # Move to target position
        print("\n" + "-" * 50)
        print("Moving to target position...")
        move_robot_to_position(robot, TARGET_POS)
        
        # Get final position
        print("\n" + "-" * 50)
        print("Final Position:")
        get_current_position(robot)
        
        print("\n" + "=" * 50)
        print("Script completed successfully!")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
    finally:
        # Always close the robot connection
        print("\nClosing robot connection...")
        try:
            robot.close()
            print("Robot connection closed.")
        except:
            pass


if __name__ == "__main__":
    main()

