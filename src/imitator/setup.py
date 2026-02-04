import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'imitator'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='erik',
    maintainer_email='erik.helmut1@gmail.com',
    description='Imitator ROS2 Jazzy package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'force_control = imitator.force_control:main',
            'imitator_node = imitator.imitator_node:main',
        ],
    },
)
