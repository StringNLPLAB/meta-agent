from setuptools import setup, find_packages

setup(
    name="colbenchgym",
    version="0.1.0",
    description="A Gymnasium environment for Collaborative Agent Bench (ColBench) tasks",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ]
    },
    include_package_data=True,
    package_data={
        "colbenchgym": ["prompts/*.txt"],
    },
)
