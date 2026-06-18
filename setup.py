"""
YOLOv8目标检测系统 - 安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 读取依赖
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name='yolov8-object-detection',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='基于YOLOv8的智能目标检测系统',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/yolov8-object-detection',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'flake8>=3.9.0',
            'black>=21.0.0',
            'isort>=5.9.0',
        ],
        'docs': [
            'mkdocs>=1.2.0',
            'mkdocs-material>=7.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'detect=examples.detect_image:main',
        ],
    },
)
