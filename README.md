# Code for Articles/Blog Posts

## Purpose

All code for the articles posted on my [personal website](https://jacob-pieniazek.com/articles)

The code for the following articles is included:

1. **Controlling for "X"?** (`fwl.py`)
1. **Predictive Parameters in a Logistic Regression: Making Sense of it All** (`logistic.py`)
1. **Optimization, Newton’s Method, & Profit Maximization: Pt. 1, Pt. 2, & Pt. 3** (`nm1.py`, `nm2.py`, `nm3.py`)
1. **t-SNE from Scratch (ft. NumPy)** (`tsne.py`)
1. **Double Machine Learning Simplified: Part 1 & Part 2** (`dml1.py`, `dml2.py`)

## Environment

These articles were developed using [Marimo](https://marimo.io/). We also provide jupyter notebooks for those who prefer that, available in the `jupyter` directory, but Marimo is recommended as jupyter notebooks may contain bugs, since we convert them from Marimo.

1. Clone repo to local machine 
1. Install [uv](https://docs.astral.sh/uv/)
1. Create environment via `uv sync --frozen`    

### Marimo

1. Run `uv sync --group marimo --frozen`
1. Run `marimo edit` to open the Marimo editor and navigate to the article you want to interact with under `marimo/`

### Jupyter

1. Run `uv sync --group jupyter --frozen`
1. Run `jupyter notebook` to open the jupyter notebook and navigate to the article you want to interact with under `jupyter/`

