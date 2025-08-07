from setuptools import setup

package_name = 'oculus_replay_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    py_modules=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Stan Schultes',
    maintainer_email='stanschultes@live.com',
    description='Replay .oculus sonar data as PointCloud2',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'oculus_replay_node = oculus_replay_node.oculus_replay_node:main',
            'oculus_file_reader_node = oculus_replay_node.oculus_file_reader_node:main',
        ],
    },
)
