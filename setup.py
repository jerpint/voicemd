from setuptools import setup, find_packages


setup(
    name='voicemd',
    version='0.0.1',
    packages=find_packages(include=['voicemd', 'voicemd.*']),
    python_requires='>=3.9',
    install_requires=[
        'librosa==0.10.0.post2',
        'mlflow==2.4.1',
        'numpy==1.24.3',
        'torch==2.0.1',
        'torchaudio==2.0.2',
        'torchvision==0.15.2',
    ],
    entry_points={
        'console_scripts': [
            'main=voicemd.main:main'
        ],
    }
)
