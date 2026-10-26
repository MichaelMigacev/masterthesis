import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="domip", # Replace with your own username
    version="1.0.0",
    author="Vito Mengers",
    author_email="v.mengers@tu-berlin.de",
    description="Differentiable Online Multimodal Interactive Perception",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.tu-berlin.de/rbo/robotics/domip",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[]
)
