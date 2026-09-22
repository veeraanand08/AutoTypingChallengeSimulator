from setuptools import setup

package_name = 'autotyper'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/assets', ['assets/keyboard_grid.png']),
        ('share/' + package_name + '/launch', ['launch/typist.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team 6',
    maintainer_email='team6@example.com',
    description='Autonomous typing node for the URC autotyping simulator',
    license='MIT',
    entry_points={'console_scripts': ['typist = autotyper.typist_node:main']},
)
