from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_share_directory
from launch.actions.include_launch_description import IncludeLaunchDescription
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, FindExecutable
from launch.launch_description_sources.python_launch_description_source import PythonLaunchDescriptionSource


def launch_setup(context, *args, **kwargs):
    recycle = Node(
        package="kuka_kontrol",
        executable="recycle.py",
        name="recycle",
    )

    recycle_vision = Node(
        package="revision",
        executable="revision_service.py",
        name="recycle_vision",
    )

    to_start = [recycle, recycle_vision]

    return to_start


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])
