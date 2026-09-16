import os
from glob import glob
from setuptools import find_packages, setup

package_name = "scene_graph_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yash Bhardwaj",
    maintainer_email="user@example.com",
    description="ROS 2 transport layer for Dynamic 3D Scene Graph",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "scene_graph_node = scene_graph_ros.scene_graph_node:main",
            "tum_player = scene_graph_ros.tum_player:main",
            "live_2d_viewer = scene_graph_ros.live_2d_viewer:main",
        ],
    },
)
