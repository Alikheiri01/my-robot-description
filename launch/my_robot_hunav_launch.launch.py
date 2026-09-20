"""
Combined launch, REVISED after confirming the Gazebo-actor pipeline
(WorldGenerator -> HuNavSystemPluginIGN -> SDF <actor>) hits an upstream
Ignition Gazebo rendering bug (Ogre::ItemIdentityException on actor
double-registration once depth-camera sensors trigger a second render
pass -- same bug class as gazebosim/gz-sim#1489). Confirmed via two
independent experiments (shortening the spawn delay, switching render
backends) that this is not fixable from our side without eliminating the
second registration pass entirely.

NEW ARCHITECTURE: Gazebo never loads WorldGenerator's output or any SDF
<actor> at all. It loads the stock empty.sdf (same as the original
robot-only launch), with the robot AND a plain "pedestrian_standin" box
model (immune to the actor bug -- plain models already proven to survive
the same duplicate-registration condition that crashed actors) spawned
into it normally. hunav_loader + hunav_gazebo_world_generator +
hunav_agent_manager still run, headless, purely so a new bridge node
(hunav_model_bridge.py) can use REAL Social Force Model computation
(/get_agents once for initial state, /compute_agents every cycle) to move
the plain model -- see that node's own docstring for the full design.

robot_name MUST still match spawn_entity's -name below, even though
HuNavSystemPluginIGN never actually runs now -- hunav_gazebo_world_generator
still declares/reads this parameter unconditionally in its own Configure(),
even though its Gazebo-facing output is discarded.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, RegisterEventHandler, DeclareLaunchArgument,
    AppendEnvironmentVariable, LogInfo, TimerAction, ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessStart
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_name = 'my_robot_description'
    pkg_share = get_package_share_directory(pkg_name)

    # ---- Launch arguments ----
    environment_name = LaunchConfiguration('environment_name')
    configuration_file = LaunchConfiguration('configuration_file')
    robot_name = LaunchConfiguration('robot_name')
    update_rate = LaunchConfiguration('update_rate')
    use_gazebo_obs = LaunchConfiguration('use_gazebo_obs')
    global_frame = LaunchConfiguration('global_frame_to_publish')
    use_navgoal = LaunchConfiguration('use_navgoal_to_start')
    ignore_models = LaunchConfiguration('ignore_models')
    plugin_position = LaunchConfiguration('plugin_position')

    declare_args = [
        # NOTE: environment_name is still required by hunav_gazebo_world_generator's
        # own Configure() -- it still processes a base world file even though its
        # OUTPUT (generatedWorld.sdf) is never loaded by Gazebo anymore. Reusing
        # thesis_base.sdf here is harmless; nothing about Gazebo's own world
        # depends on it.
        DeclareLaunchArgument('environment_name', default_value='thesis_base'),
        DeclareLaunchArgument('configuration_file', default_value='thesis_static_agent.yaml'),
        DeclareLaunchArgument('robot_name', default_value='my_robot',
                               description='MUST match the -name given to spawn_entity below'),
        DeclareLaunchArgument('update_rate', default_value='30.0'),
        DeclareLaunchArgument('use_gazebo_obs', default_value='true'),
        DeclareLaunchArgument('global_frame_to_publish', default_value='map'),
        DeclareLaunchArgument('use_navgoal_to_start', default_value='false'),
        DeclareLaunchArgument('ignore_models', default_value='ground_plane'),
        DeclareLaunchArgument('plugin_position', default_value='0'),
    ]

    # =========================================================================
    # HuNav headless pieces -- used ONLY for /get_agents and /compute_agents.
    # world_generator's own Gazebo-world-generation output is never consumed.
    # =========================================================================

    world_file = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_fortress_wrapper'), 'worlds',
        PythonExpression(["'", environment_name, ".sdf'"])
    ])
    agent_conf_file = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_fortress_wrapper'), 'scenarios', configuration_file
    ])

    hunav_models_path = PathJoinSubstitution([FindPackageShare('hunav_gazebo_fortress_wrapper'), 'worlds'])
    set_env_gz_resources = AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', hunav_models_path)
    set_env_gazebo_resources = AppendEnvironmentVariable('GAZEBO_RESOURCE_PATH', hunav_models_path)
    robot_models_path = os.path.join(pkg_share, '..')
    set_env_robot_resources = AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', robot_models_path)

    hunav_loader_node = Node(
        package='hunav_agent_manager', executable='hunav_loader',
        output='screen', parameters=[agent_conf_file],
    )
    hunav_worldgen_node = Node(
        package='hunav_gazebo_fortress_wrapper', executable='hunav_gazebo_world_generator',
        output='screen',
        parameters=[{
            'base_world': world_file, 'use_gazebo_obs': use_gazebo_obs, 'update_rate': update_rate,
            'robot_name': robot_name, 'global_frame_to_publish': global_frame,
            'use_navgoal_to_start': use_navgoal, 'ignore_models': ignore_models,
            'plugin_position': plugin_position,
        }],
    )
    hunav_manager_node = Node(
        package='hunav_agent_manager', executable='hunav_agent_manager',
        name='hunav_agent_manager', output='screen', parameters=[{'use_sim_time': True}],
    )
    static_tf_node = Node(
        package='tf2_ros', executable='static_transform_publisher', output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    ordered_worldgen = RegisterEventHandler(OnProcessStart(
        target_action=hunav_loader_node,
        on_start=[LogInfo(msg='hunav_loader started, launching world generator after 2s...'),
                  TimerAction(period=2.0, actions=[hunav_worldgen_node])],
    ))

    # =========================================================================
    # Gazebo -- stock empty.sdf, exactly like the original robot-only launch.
    # No custom world, no WorldGenerator output, no SDF <actor> anywhere.
    # =========================================================================

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # =========================================================================
    # Robot + plain pedestrian stand-in + existing depth/occupancy pipeline
    # =========================================================================

    xacro_file = os.path.join(pkg_share, 'models', 'my_robot_description', 'my_robot.urdf.xacro')
    robot_description = {'robot_description': ParameterValue(Command(['xacro ', xacro_file]), value_type=str)}
    node_robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen', parameters=[robot_description, {'use_sim_time': True}],
    )
    spawn_entity = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-topic', 'robot_description', '-name', robot_name,
                   '-x', '0.0', '-y', '0.0', '-z', '2.0', '-Y', '0.7854'],
        output='screen',
    )

    # Plain box model, immune to the actor-rendering bug -- see
    # hunav_assets/worlds/pedestrian_standin.sdf's own comments.
    pedestrian_standin_sdf = os.path.join(
        pkg_share, 'hunav_assets', 'worlds', 'pedestrian_standin.sdf')
    spawn_pedestrian_standin = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-file', pedestrian_standin_sdf, '-name', 'pedestrian_standin',
                   '-x', '1.4', '-y', '1.4', '-z', '0.8', '-Y', '-2.356'],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState@ignition.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/depth_cam/left/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/left/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/left/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/depth_cam/left/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            '/depth_cam/right/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/right/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/depth_cam/right/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/depth_cam/right/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
        ],
        output='screen',
    )
    left_camera_tf_node = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'left_camera_link', '--child-frame-id', 'my_robot/base_link/left_camera'],
    )
    right_camera_tf_node = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0', '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'right_camera_link', '--child-frame-id', 'my_robot/base_link/right_camera'],
    )
    camera_info_fixer_process = ExecuteProcess(
        cmd=['python3', '/home/ali/ros2_ws/src/my_robot_description/codes/camera_info_fixer.py'],
        output='screen',
    )
    depth_proc_container = ComposableNodeContainer(
        name='depth_image_proc_container', namespace='', package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(package='depth_image_proc', plugin='depth_image_proc::PointCloudXyzNode',
                            name='point_cloud_xyz_left',
                            remappings=[('image_rect', '/depth_cam/left/depth_image'),
                                        ('camera_info', '/depth_cam/left/camera_info_fixed'),
                                        ('points', '/depth_cam/left/points_corrected')]),
            ComposableNode(package='depth_image_proc', plugin='depth_image_proc::PointCloudXyzNode',
                            name='point_cloud_xyz_right',
                            remappings=[('image_rect', '/depth_cam/right/depth_image'),
                                        ('camera_info', '/depth_cam/right/camera_info_fixed'),
                                        ('points', '/depth_cam/right/points_corrected')]),
        ],
        output='screen',
    )
    depth_fusion_process = ExecuteProcess(
        cmd=['python3', '/home/ali/ros2_ws/src/my_robot_description/codes/fuse_depth_clouds.py'],
        output='screen',
    )

    robot_and_pipeline_actions = [
        node_robot_state_publisher, spawn_entity, spawn_pedestrian_standin, bridge,
        left_camera_tf_node, right_camera_tf_node,
        camera_info_fixer_process, depth_proc_container, depth_fusion_process,
    ]
    # Gazebo (stock empty.sdf) starts fast with nothing to generate first --
    # a couple of seconds is plenty, unlike the old chained-off-worldgen timing.
    delayed_robot_spawn = TimerAction(period=3.0, actions=robot_and_pipeline_actions)

    # Bridge node starts after the robot + stand-in model + HuNav headless
    # pieces have all had a chance to come up -- needs /odom publishing,
    # pedestrian_standin existing in Gazebo, and /get_agents + /compute_agents
    # both servable.
    hunav_model_bridge_process = ExecuteProcess(
        cmd=['python3', '/home/ali/ros2_ws/src/my_robot_description/hunav_codes/hunav_model_bridge.py'],
        output='screen',
    )
    # heading_smoother.py just subscribes to /people (published by the
    # bridge above) -- no strict ordering needed between the two, ROS2
    # subscriptions connect whenever both sides are up regardless of start
    # order, so it's fine to start them together.
    heading_smoother_process = ExecuteProcess(
        cmd=['python3', '/home/ali/ros2_ws/src/my_robot_description/hunav_codes/heading_smoother.py'],
        output='screen',
    )
    delayed_bridge_start = TimerAction(
        period=8.0, actions=[hunav_model_bridge_process, heading_smoother_process])

    ld = LaunchDescription()
    for a in declare_args:
        ld.add_action(a)
    ld.add_action(set_env_gz_resources)
    ld.add_action(set_env_gazebo_resources)
    ld.add_action(set_env_robot_resources)
    ld.add_action(static_tf_node)
    ld.add_action(hunav_loader_node)
    ld.add_action(ordered_worldgen)
    ld.add_action(hunav_manager_node)
    ld.add_action(gazebo)
    ld.add_action(delayed_robot_spawn)
    ld.add_action(delayed_bridge_start)
    return ld