from setuptools import setup
import re

with open("README.md", "r") as f:
    desc = f.read()
    desc = desc.split("<!-- content -->")[-1]
    desc = re.sub("<[^<]+?>", "", desc)  # Remove html


setup(
    name="PyPortfolioOpt",
    version="0.2.0",
    description="Financial portfolio optimisation in python",
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/robertmartin8/PyPortfolioOpt",
    author="Robert Andrew Martin",
    author_email="martin.robertandrew@gmail.com",
    license="MIT",
    packages=["pypfopt"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="portfolio finance optimization quant trading investing",
    install_requires=["numpy", "pandas", "scikit-learn", "scipy", "noisyopt"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    python_requires=">=3",
    project_urls={
        "Documentation": "https://pyportfolioopt.readthedocs.io/en/latest/",
        "Issues": "https://github.com/robertmartin8/PyPortfolioOpt/issues",
        "Personal website": "https://reasonabledeviations.science",
    },
)
