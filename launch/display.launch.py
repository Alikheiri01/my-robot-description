from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('my_robot_description')
    
    # Path to the Xacro file
    xacro_file = os.path.join(pkg_share, 'models', 'my_robot_description', 'my_robot.urdf.xacro')
    
    # Path to the RViz config file (Make sure the folder name 'rviz' matches where you saved it)
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'my_robot_config.rviz')

    robot_description_content = Command(['xacro ', xacro_file])
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[robot_description] 
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            # This line tells RViz to load your saved config
            arguments=['-d', rviz_config_file]
        ),
    ])