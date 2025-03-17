from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'go2_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alexlin',
    maintainer_email='alex.lin416@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_image = go2_detection.detect_image:main',
            'pc_to_depth = go2_detection.pc_to_depth:main',
            'csi_camera_pub = go2_detection.csi_camera_pub:main',
        ],
    },
)
