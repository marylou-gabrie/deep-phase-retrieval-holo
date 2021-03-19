import setuptools


long_description="""
Python package for phase retrieval research
"""

setuptools.setup(
    name="phase_retrieval_code_demo", 
    version="0.0.1",
    author="Marylou Gabrié, Hannah Lawrence, Michael Eickenberg, Henry Li, David A. Barmherzig,",
    author_email="mgabrie@nyu.edu",
    description="python package for phase retrieval research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
