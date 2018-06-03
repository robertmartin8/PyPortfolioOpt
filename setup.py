from setuptools import setup

desc = """PyPortfolioOpt offers an intuitive, well-documented solution to various portfolio optimisation problems
that is both extensive and extensible. Currently there is support for maximum-sharpe, minimum-volatility,
efficient risk (max return for a given risk) and efficient return (min risk for a target return). Classical
and experimental options are provided, including better estimates of covariance."""


setup(
    name="PyPortfolioOpt",
    version="0.1.0",
    description="PyPortfolioOpt: Efficient Frontier portfolio optimisation",
    long_description=desc,
    url="https://github.com/robertmartin8/PyPortfolioOpt",
    author="Robert Andrew Martin",
    author_email="martin.robertandrew@gmail.com",
    license="MIT",
    packages=["pypfopt"],
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    keywords="portfolio finance optimization quant trading investing",
    install_requires=["numpy", "pandas", "pytest", "scikit-learn", "scipy"],
    project_urls=(
        {  # TODO add documentation link
            "Issues?": "https://github.com/robertmartin8/PyPortfolioOpt/issues",
            "Personal website": "https://reasonabledeviations.science",
        },
    ),
)
