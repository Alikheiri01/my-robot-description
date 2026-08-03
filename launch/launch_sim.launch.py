import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable 
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg_name = 'my_robot_description'
    pkg_share = get_package_share_directory(pkg_name)

    # 1. SETUP GAZEBO RESOURCE PATH
    ros_gz_resource_path = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_share, '..')
    )

    # 2. PROCESS THE URDF
    xacro_file = os.path.join(pkg_share, 'models', 'my_robot_description', 'my_robot.urdf.xacro')
    robot_description_content = Command(['xacro ', xacro_file])
    
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    # 3. ROBOT STATE PUBLISHER
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description, 
            {'use_sim_time': True}
        ]
    )

    # 4. START GAZEBO (IGNITION)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # 5. SPAWN THE ROBOT
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description',
                   '-name', 'my_robot',
                   '-x', '0.0',
                   '-y', '0.0',
                   '-z', '2.0',
                   '-Y', '0.7854'],
        output='screen'
    )

    # 6. THE BRIDGE
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Base kinematics and state
            '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState@ignition.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            
            # Left RGB-D Camera
            '/depth_cam/left/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/left/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/left/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/depth_cam/left/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            
            # Right RGB-D Camera
            '/depth_cam/right/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/right/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/right/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/depth_cam/right/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked'
        ],
        output='screen'
    )

    # 7. STATIC TRANSFORMS (Connects Gazebo internal camera frames to URDF optical frames)
    left_camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'my_robot/base_link/left_camera', 'left_camera_link_optical']
    )

    right_camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'my_robot/base_link/right_camera', 'right_camera_link_optical']
    )

    return LaunchDescription([
        ros_gz_resource_path,
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        bridge,
        left_camera_tf_node,
        right_camera_tf_node,
    ])