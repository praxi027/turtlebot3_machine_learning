from setuptools import find_packages
from setuptools import setup

package_name = 'turtlebot3_ppo'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'numpy',
    ],
    zip_safe=True,
    author='User',
    author_email='user@example.com',
    maintainer='User',
    maintainer_email='user@example.com',
    description='TurtleBot3 PPO reinforcement learning with continuous action space',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ppo_agent       = turtlebot3_ppo.ppo_agent:main',
            'ppo_environment = turtlebot3_ppo.ppo_environment:main',
            'ppo_gazebo      = turtlebot3_ppo.ppo_gazebo:main',
            'result_graph    = turtlebot3_ppo.result_graph:main',
        ],
    },
)
