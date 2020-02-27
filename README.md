
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

### Set Google credentials

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials/json
``` 

### Run demo notebook

```bash
jupyter-notebook demo[-remote].ipynb
```

## Notebooks structure

### demo

#### Part I. Usage mlpanel from notebook

Demonstrates how to communicate with tracking server running by **mlpanel** 
directly from Jupyter notebook.

In this part first function which used in ML-workflow are defined. The main function is
**run_experiment()** - creates new run on tracking server.

Part one has screenshots of UI usage: view projects, experiments, runs, register models and 
create deployments.

#### Part II. Usage mlpanel from code

There is ML project example - **IrisProject**. It's MLflow project ready to use.  
In Part II **IrisProject** logs parameters, metrics, artifacts and models to tracking server
running also by **mlpanel**. 

---

Both parts shows that it's not difference where from to communicate with **mlpanel** - 
main you must specify MLflow tracking uri for you project and run it.


### demo-remote
 
#### Part I and Part II

Like in demo

### Part III

About how to predict data