"""
Robot Controller for URBasic Integration
Manages robot connection and provides movement functions for the agent.
"""

import sys
import os
import math
import time

# Add parent directory to path to import URBasic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import URBasic.robotModel
import URBasic.urScriptExt


class RobotController:
    """Controller class for managing UR robot connection and movements."""
    
    def __init__(self, robot_ip='169.254.152.222', accel=0.5, vel=0.6):
        """
        Initialize the robot controller.
        
        Args:
            robot_ip: IP address of the robot
            accel: Default acceleration (rad/s^2)
            vel: Default velocity (rad/s)
        """
        self.robot_ip = robot_ip
        self.accel = accel
        self.vel = vel
        self.robot = None
        self.robot_model = None
        self.connected = False
        
        # Home position (in radians) - adjust as needed
        self.HOME_POS = (
            math.radians(-90),
            math.radians(-90),
            math.radians(90),
            math.radians(-90),
            math.radians(-90),
            math.radians(90)
        )
    
    def connect(self):
        """Connect to the robot."""
        if self.connected:
            return True
        
        if not URBasic_available:
            print("Warning: URBasic is not available. Using mock robot connection.")
            print("To enable real robot control, please ensure URBasic is properly installed.")
            self.connected = True  # Allow mock connection for testing
            return True
        
        try:
            print(f"Connecting to robot at {self.robot_ip}...")
            self.robot_model = URBasic.robotModel.RobotModel()
            self.robot = URBasic.urScriptExt.UrScriptExt(
                host=self.robot_ip, 
                robotModel=self.robot_model
            )
            self.robot.reset_error()
            self.robot.init_realtime_control()
            time.sleep(1)
            self.connected = True
            print("Robot connected successfully!")
            return True
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the robot."""
        if self.robot and self.connected:
            try:
                self.robot.close()
                self.connected = False
                print("Robot disconnected.")
            except Exception as e:
                print(f"Error disconnecting: {e}")
    
    def move_to_position(self, joint_angles, accel=None, vel=None):
        """
        Move robot to specified joint angles.
        
        Args:
            joint_angles: List or tuple of 6 joint angles in degrees
            accel: Acceleration (rad/s^2), uses default if None
            vel: Velocity (rad/s), uses default if None
        
        Returns:
            str: Success or error message
        """
        if not self.connected:
            if not self.connect():
                return "Error: Could not connect to robot"
        
        try:
            # Convert degrees to radians if needed
            if isinstance(joint_angles, (list, tuple)):
                if len(joint_angles) != 6:
                    return "Error: Must provide exactly 6 joint angles"
                # Check if values are in degrees (likely > 2*pi) or radians
                if any(abs(angle) > 2 * math.pi for angle in joint_angles):
                    # Assume degrees, convert to radians
                    joint_angles = tuple(math.radians(angle) for angle in joint_angles)
                else:
                    joint_angles = tuple(joint_angles)
            else:
                return "Error: joint_angles must be a list or tuple"
            
            accel = accel or self.accel
            vel = vel or self.vel
            
            print(f"Moving robot to position: {[math.degrees(a) for a in joint_angles]} degrees")
            self.robot.movej(q=joint_angles, a=accel, v=vel)
            time.sleep(1)  # Wait for movement to complete
            return f"Successfully moved to position: {[round(math.degrees(a), 2) for a in joint_angles]} degrees"
        except Exception as e:
            return f"Error moving robot: {str(e)}"
    
    def move_to_home(self):
        """
        Move robot to home position.
        
        Returns:
            str: Success or error message
        """
        return self.move_to_position(self.HOME_POS)
    
    def get_current_position(self):
        """
        Get current joint positions.
        
        Returns:
            str: Current position in degrees or error message
        """
        if not self.connected:
            if not self.connect():
                return "Error: Could not connect to robot"
        
        try:
            joints = self.robot.get_actual_joint_positions()
            degrees = [round(math.degrees(j), 2) for j in joints]
            return f"Current position: {degrees} degrees"
        except Exception as e:
            return f"Error getting position: {str(e)}"
    
    def set_home_position(self, joint_angles):
        """
        Set a new home position.
        
        Args:
            joint_angles: List or tuple of 6 joint angles in degrees
        """
        if isinstance(joint_angles, (list, tuple)) and len(joint_angles) == 6:
            if any(abs(angle) > 2 * math.pi for angle in joint_angles):
                self.HOME_POS = tuple(math.radians(angle) for angle in joint_angles)
            else:
                self.HOME_POS = tuple(joint_angles)
            return f"Home position updated to: {[round(math.degrees(a), 2) for a in self.HOME_POS]} degrees"
        return "Error: Must provide exactly 6 joint angles"


# Global robot controller instance
_robot_controller = None


def get_robot_controller(robot_ip='169.254.152.222'):
    """Get or create the global robot controller instance."""
    global _robot_controller
    if _robot_controller is None:
        _robot_controller = RobotController(robot_ip=robot_ip)
    return _robot_controller

