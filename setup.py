from setuptools import setup, find_packages


setup(
    name="voicemd",
    version="0.0.1",
    packages=find_packages(include=["voicemd", "voicemd.*"]),
    python_requires=">=3.7",
    install_requires=[],
    extras_require={
        'dev': [
            "flake8",
            "pyyaml",
            "torch",
            "librosa>=0.8",
            "torchaudio",
            "tqdm",
            "mlflow",
            "orion",
            "pytest",
        ]
    },
    entry_points={
        "console_scripts": ["main=voicemd.main:main"],
    },
)
