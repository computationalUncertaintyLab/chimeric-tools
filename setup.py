import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chimeric-tools",
    version="0.0.4",
    author="Abraham Berlin, Wenxuan Ye",
    author_email="alb323@lehgih.edu, lehighwenxuanye@gmail.com",
    description="ComputationalUncertaintyLab/chimeric-tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/computationalUncertaintyLab/chimeric-tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    include_package_data=True,
    package_data={"": ["data/*.csv"]},
)
