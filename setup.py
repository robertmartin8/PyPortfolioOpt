from distutils.core import setup

with open("README.rst") as f:
    readme = f.read()

setup(
    name="PyPortfolioOpt",
    version="0.1",
    description="PyPortfolioOpt: Efficient Frontier, Black Litterman, Monte Carlo optimisation methods",
    long_description=readme,
    author="Robert Andrew Martin",
    author_email="martin.robertandrew @ gmail.com",
    packages=["pypfopt", "pypfopt.tests"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
