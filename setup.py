import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

def load_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]

requirements = load_requirements()

__version__ = "0.0.1"

REPO_NAME = "EcoClassify---Wildlife-Image-Classifier"
AUTHOR_USER_NAME = "santosh3110"
SRC_REPO = "ecoclassify"
AUTHOR_EMAIL = "santoshkumarguntupalli@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A modular wildlife image classifier using CNN and transfer learning (ResNet)",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
