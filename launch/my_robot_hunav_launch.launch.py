"""
Combined launch: HuNav generates a merged world (thesis_base.sdf + agents +
HuNavPlugin) -> Gazebo launches on that generated world -> the thesis robot
spawns into the SAME running instance -> the existing depth/occupancy
pipeline (bridge, camera TF, camera_info_fixer, depth fusion) starts exactly
as it already did in the original robot-only launch file.

Nothing about the original robot-only launch file changes -- it still works
standalone. This is a new, separate launch file for the combined scenario.

TIMING NOTE (matches the same fragile-but-already-established pattern
upstream's own simulation_fortress.launch.py uses -- fixed-delay TimerActions
rather than a proper readiness check): the delays below are first guesses,
not measured. If the robot fails to spawn or the bridge starts before Gazebo
has finished loading the generated world, lengthen the relevant TimerAction
period rather than assume something else is wrong -- start there before
treating it as a new bug. HuNavSystemPluginIGN's own robot-lookup already
retries every frame until the robot entity appears (confirmed from its
source), so a robot spawned a bit late should still be found correctly;
it's the *pipeline* nodes (bridge, camera TF, etc.) that have no such retry
and need Gazebo to genuinely be up first.

robot_name MUST match the actual -name given to spawn_entity below (my_robot)
-- HuNavSystemPluginIGN looks up the robot by this exact Gazebo model name.
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

    # ---- Launch arguments -- our own sensible defaults, not the demo's ----
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
        DeclareLaunchArgument('environment_name', default_value='thesis_base',
                               description='Base world file name (no .sdf), looked up in '
                                            'hunav_gazebo_fortress_wrapper/worlds/ -- must be '
                                            'symlinked there from my_robot_description/hunav_assets/worlds/'),
        DeclareLaunchArgument('configuration_file', default_value='thesis_static_agent.yaml',
                               description='Agent scenario YAML, looked up in '
                                            'hunav_gazebo_fortress_wrapper/scenarios/ -- must be '
                                            'symlinked there from my_robot_description/hunav_assets/scenarios/'),
        DeclareLaunchArgument('robot_name', default_value='my_robot',
                               description='MUST match the -name given to spawn_entity below'),
        DeclareLaunchArgument('update_rate', default_value='30.0',
                               description='HuNavPlugin update rate (Hz) -- 30.0 confirmed to fix '
                                            'jitter/RTF instability seen at the upstream default of 1000.0'),
        DeclareLaunchArgument('use_gazebo_obs', default_value='true'),
        DeclareLaunchArgument('global_frame_to_publish', default_value='map'),
        DeclareLaunchArgument('use_navgoal_to_start', default_value='false'),
        DeclareLaunchArgument('ignore_models', default_value='ground_plane',
                               description='Space-separated Gazebo model names agents should not '
                                            'treat as obstacles -- ground_plane matches thesis_base.sdf; '
                                            'add the robot\'s own link names here later if agents react '
                                            'oddly to the robot\'s body via the generic obstacle path'),
        DeclareLaunchArgument('plugin_position', default_value='0'),
    ]

    # =========================================================================
    # HuNav world generation + Gazebo (adapted from simulation_fortress.launch.py)
    # =========================================================================

    world_file = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_fortress_wrapper'), 'worlds',
        PythonExpression(["'", environment_name, ".sdf'"])
    ])
    agent_conf_file = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_fortress_wrapper'), 'scenarios', configuration_file
    ])
    generated_world = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_fortress_wrapper'), 'worlds', 'generatedWorld.sdf'
    ])

    # NOTE: written using AppendEnvironmentVariable only (tolerates an unset
    # variable) -- unlike simulation_fortress.launch.py's own env setup, which
    # crashes outright if GZ_SIM_RESOURCE_PATH/GAZEBO_RESOURCE_PATH aren't
    # already exported first. Fixed here since this is our own launch file;
    # no "export ... ''" prerequisite needed before running this one.
    hunav_models_path = PathJoinSubstitution([FindPackageShare('hunav_gazebo_fortress_wrapper'), 'worlds'])
    set_env_gz_resources = AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', hunav_models_path)
    set_env_gazebo_resources = AppendEnvironmentVariable('GAZEBO_RESOURCE_PATH', hunav_models_path)
    # Robot's own mesh/model resource path, from the original robot-only launch file
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
    # Identity map->odom, matching HuNav's own default assumption -- same
    # explicit upstream comment applies: remove this if/when the robot ever
    # gets a real localization stack.
    static_tf_node = Node(
        package='tf2_ros', executable='static_transform_publisher', output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    ordered_worldgen = RegisterEventHandler(OnProcessStart(
        target_action=hunav_loader_node,
        on_start=[LogInfo(msg='hunav_loader started, launching world generator after 2s...'),
                  TimerAction(period=2.0, actions=[hunav_worldgen_node])],
    ))

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r -s -v4 ', generated_world], 'on_exit_shutdown': 'true'}.items(),
    )
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v4'}.items(),
    )
    ordered_gazebo = RegisterEventHandler(OnProcessStart(
        target_action=hunav_worldgen_node,
        on_start=[LogInfo(msg='World generator started, launching Gazebo after 3s...'),
                  TimerAction(period=3.0, actions=[gzserver_cmd, gzclient_cmd])],
    ))

    # =========================================================================
    # Robot + existing depth/occupancy pipeline (from the original robot-only
    # launch file, unchanged, just re-timed to start after Gazebo is up)
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
    # Same hardcoded absolute paths as the original launch file -- unrelated
    # pre-existing cleanup item, out of scope for this file.
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
        node_robot_state_publisher, spawn_entity, bridge,
        left_camera_tf_node, right_camera_tf_node,
        camera_info_fixer_process, depth_proc_container, depth_fusion_process,
    ]
    ordered_robot_spawn = RegisterEventHandler(OnProcessStart(
        target_action=gzserver_cmd,
        on_start=[LogInfo(msg='Gazebo started, spawning robot + pipeline after 5s...'),
                  TimerAction(period=5.0, actions=robot_and_pipeline_actions)],
    ))

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
    ld.add_action(ordered_gazebo)
    ld.add_action(ordered_robot_spawn)
    return ld