from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'vocal_command_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # (
        #     os.path.join('share', package_name, 'Vosk_Small/am'),
        #     ['Vosk_Small/am/final.mdl']
        # ),
        # (
        #     os.path.join('share', package_name, 'Vosk_Small/conf'),
        #     ['Vosk_Small/conf/mfcc.conf']
        # ),
        # (
        #     os.path.join('share', package_name, 'Vosk_Small'),
        #     glob('Vosk_Small/**/*', recursive=True)
        # ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xplore',
    maintainer_email='xplore@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vocom_node = vocal_command_pkg.node:main',
            'fake_cs_node = vocal_command_pkg.fake_cs_node:main',
        ],
    },
)
