"""
Combined launch -- ACTOR VERSION.

Kept as a NEW file rather than an edit to my_robot_hunav_launch.launch.py,
which stays untouched as the validated plain-model baseline. Same
convention already used for build_occupancy_grid_dynamic.py versus
build_occupancy_grid.py.

WHY THIS EXISTS
The baseline launch spawns a plain MODEL ("pedestrian_standin") instead
of an SDF actor, because every earlier attempt to have an actor in the
scene crashed Gazebo with Ogre::ItemIdentityException as soon as the
robot's depth cameras triggered their first render pass (same bug class
as gazebosim/gz-sim#1489). That workaround cost us real skeletal
animation: a plain model has no skeleton, so the pedestrian slid around
frozen in its bind pose with its arms out.

Confirmed by experiment (2026-09-21) that the crash is an ORDERING
problem, not an actor problem. The render engine collides with an actor
that ALREADY EXISTS when the sensors first initialize. Spawning the
actor dynamically, after the robot's sensors are up and have rendered at
least one frame, avoids it entirely -- verified by hand: no crash, the
actor renders and animates normally alongside my_robot and its cameras.

Also confirmed: real walking needs NO C++ plugin work. The actor's
interpolate_x flag makes Gazebo tie walk-cycle playback to the actor's
measured displacement, so the ordinary set_pose calls that
hunav_model_bridge.py already makes produce genuine walking motion.
That node is therefore UNCHANGED by this work -- it simply targets a
different entity name, which it reads from hunav_config.py.

TIMING -- the one fragile part. Tune here if the Ogre crash ever returns.
    t=3s   robot, cameras and depth pipeline spawn
    t=6s   pedestrian actor spawns (MUST be after sensors have rendered)
    t=8s   hunav_model_bridge and heading_smoother start driving it
These are fixed delays, not event-driven guarantees. On a slower machine
the actor could still land before the sensor render pass and bring the
old crash back. A robust version would trigger the actor spawn off the
first message on /depth_cam/left/depth_image instead of a wall clock.
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


# =============================================================================
# Actor identity, spawn pose and timing -- the values most likely to need
# tuning, deliberately grouped here instead of buried in the body.
#
# ACTOR_NAME MUST match MODEL_NAME in hunav_config.py -- that is the name
# hunav_model_bridge.py hands to set_pose on every cycle, and it must also
# match the actor name inside hunav_actor.sdf itself.
#
# ACTOR_SPAWN_X/Y/YAW match thesis_static_agent.yaml's agent1 init_pose, so
# the actor starts exactly where hunav_agent_manager already believes the
# agent is. If they disagree, the first bridge tick visibly teleports it.
#
# ACTOR_SPAWN_Z must match STANDIN_Z in hunav_config.py, since the bridge
# reapplies that same height on every set_pose call.
# =============================================================================
ACTOR_NAME = 'hunav_actor'
ACTOR_SPAWN_X = '1.4'
ACTOR_SPAWN_Y = '1.4'
ACTOR_SPAWN_Z = '0.0'
ACTOR_SPAWN_YAW = '-2.356'

T_ROBOT_SPAWN = 3.0
T_ACTOR_SPAWN = 6.0
T_BRIDGE_START = 8.0


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
        # environment_name is still required by hunav_gazebo_world_generator's
        # own Configure() -- it still processes a base world file even though
        # its OUTPUT (generatedWorld.sdf) is never loaded by Gazebo. Reusing
        # thesis_base.sdf is harmless; Gazebo's own world does not depend on it.
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
    # Gazebo -- stock empty.sdf. No custom world, no WorldGenerator output,
    # and critically no actor present at world-load time.
    # =========================================================================

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # =========================================================================
    # Robot + depth/occupancy pipeline. NOTE: no pedestrian spawns here --
    # the actor is deliberately held back until after these sensors render.
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
    delayed_robot_spawn = TimerAction(period=T_ROBOT_SPAWN, actions=robot_and_pipeline_actions)

    # =========================================================================
    # Pedestrian ACTOR -- the whole point of this file. Spawned late and on
    # purpose: it must not exist when the depth cameras above perform their
    # one-time render-engine initialization, or Gazebo aborts with
    # Ogre::ItemIdentityException. See this file's header for the full story.
    # =========================================================================

    hunav_actor_sdf = os.path.join(
        pkg_share, 'hunav_assets', 'worlds', 'hunav_actor.sdf')
    spawn_hunav_actor = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-file', hunav_actor_sdf, '-name', ACTOR_NAME,
                   '-x', ACTOR_SPAWN_X, '-y', ACTOR_SPAWN_Y,
                   '-z', ACTOR_SPAWN_Z, '-Y', ACTOR_SPAWN_YAW],
        output='screen',
    )
    delayed_actor_spawn = TimerAction(
        period=T_ACTOR_SPAWN,
        actions=[LogInfo(msg='Sensors should be live by now -- spawning pedestrian actor.'),
                 spawn_hunav_actor])

    # =========================================================================
    # Bridge + heading smoother -- unchanged from the baseline launch.
    # Starts last: needs /odom publishing, the actor present in Gazebo, and
    # /get_agents plus /compute_agents both servable.
    # =========================================================================

    hunav_model_bridge_process = ExecuteProcess(
        cmd=['python3', '/home/ali/ros2_ws/src/my_robot_description/hunav_codes/hunav_model_bridge.py'],
        output='screen',
    )
    heading_smoother_process = ExecuteProcess(
        cmd=['python3', '/home/ali/ros2_ws/src/my_robot_description/hunav_codes/heading_smoother.py'],
        output='screen',
    )
    delayed_bridge_start = TimerAction(
        period=T_BRIDGE_START, actions=[hunav_model_bridge_process, heading_smoother_process])

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
    ld.add_action(delayed_actor_spawn)
    ld.add_action(delayed_bridge_start)
    return ld
