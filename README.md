
# Preparation

## Create virtual environment

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download dataset

```bash
wget -P data/ -nc https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
```

## Checkout to experiments branch

```bash
git checkout -b experiments
```
## Run demo

```bash
jupyter-notebook demo.ipynb
```