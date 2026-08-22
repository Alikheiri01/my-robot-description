import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
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
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'left_camera_link',
            '--child-frame-id', 'my_robot/base_link/left_camera'
        ]
    )

    right_camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'right_camera_link',
            '--child-frame-id', 'my_robot/base_link/right_camera'
        ]
    )

    # 8. CAMERA_INFO FIXER
    #    Works around a confirmed gz-sensors Fortress bug: the rgbd_camera
    #    sensor correctly applies <lens><intrinsics> to CameraInfo's K matrix,
    #    but computes the separate P (projection) matrix through a still-broken
    #    default path. depth_image_proc uses P, not K, for its 3D reconstruction,
    #    so this relay republishes corrected camera_info (both K and P) on new
    #    "_fixed" topics. Update this path if camera_info_fixer.py lives elsewhere.
    camera_info_fixer_process = ExecuteProcess(
        cmd=['python3', '/home/ali/ros2_ws/src/my_robot_description/codes/camera_info_fixer.py'],
        output='screen'
    )

    # 9. DEPTH -> POINTCLOUD CORRECTION
    #    Reconstructs the point cloud from depth_image + the corrected
    #    camera_info (from step 8) rather than trusting Gazebo's native
    #    /points topic, which has its own separate, unrelated axis bug.
    #    The original native /points topics from the bridge are left untouched.
    depth_proc_container = ComposableNodeContainer(
        name='depth_image_proc_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::PointCloudXyzNode',
                name='point_cloud_xyz_left',
                remappings=[
                    ('image_rect', '/depth_cam/left/depth_image'),
                    ('camera_info', '/depth_cam/left/camera_info_fixed'),
                    ('points', '/depth_cam/left/points_corrected'),
                ]
            ),
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::PointCloudXyzNode',
                name='point_cloud_xyz_right',
                remappings=[
                    ('image_rect', '/depth_cam/right/depth_image'),
                    ('camera_info', '/depth_cam/right/camera_info_fixed'),
                    ('points', '/depth_cam/right/points_corrected'),
                ]
            ),
        ],
        output='screen'
    )

    return LaunchDescription([
        ros_gz_resource_path,
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        bridge,
        left_camera_tf_node,
        right_camera_tf_node,
        camera_info_fixer_process,
        depth_proc_container,
    ])