from setuptools import find_packages, setup

package_name = 'neuromorphic_tactile_array'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/inference_qos.yaml']),
        ('share/' + package_name + '/config', ['config/data_collection_qos.yaml']),
    ],
    install_requires=['setuptools',
            'pyzmq',
            'msgpack',
            'msgpack-numpy',
            'opencv-python',
            'numpy',],
    zip_safe=True,
    maintainer='erik',
    maintainer_email='erik.helmut1@gmail.com',
    description='Neuromorphic Tactile Array ROS2 Jazzy Package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'nta_node = neuromorphic_tactile_array.nta_node:main',
        ],
    },
)
