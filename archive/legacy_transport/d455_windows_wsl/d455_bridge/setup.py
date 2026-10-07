from setuptools import find_packages, setup

package_name = 'd455_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yash Bhardwaj',
    maintainer_email='user@example.com',
    description='RealSense D455 Windows TCP Bridge receiver for ROS 2',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'd455_receiver = d455_bridge.d455_receiver:main',
        ],
    },
)
