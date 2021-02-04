from setuptools import setup, find_packages


setup(
    name='voicemd',
    version='0.0.1',
    packages=find_packages(include=['voicemd', 'voicemd.*']),
    python_requires='>=3.6.9',
    install_requires=[
        'flake8', 'tqdm', 'mlflow', 'orion', 'pyyaml', 'pytest',
        'torch', 'torchaudio', 'librosa>=0.8'],
    entry_points={
        'console_scripts': [
            'main=voicemd.main:main'
        ],
    }
)
