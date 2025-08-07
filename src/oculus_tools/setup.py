from setuptools import setup, find_packages

package_name = 'oculus_tools'

setup(
    name=package_name,
    version='0.1.0',
    # packages=[package_name],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/test_oculus_replay.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Stan Schultes',
    maintainer_email='stanschultes@live.com',
    description='Tools and launch files for replaying and testing Oculus sonar data',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Optional: no ROS nodes here yet
        ],
    },
)
