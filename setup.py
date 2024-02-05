from setuptools import setup, find_packages

setup(
    name="connectfour_ai",
    version="0.1.0",
    description="A Connect Four AI game implemented with Deep Q-learning.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/connectfour_ai",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pygame",
    ],
)
