# Aerospace Engineering to AI: Uncertainty Quantification in ML Classifiers

## Background and Motivation

With my background in Aerospace Engineering and Computer Science, I've always wondered if the statistical formulas and rigorous uncertainty quantification methods developed for aeronautics could be applied to machine learning. In aerospace, we can't just say "this wing design is safe"—we need to quantify _how confident_ we are, with validated statistical methods that have been tested under extreme conditions.

When I started working with ML models, I noticed something troubling: we often report "the model is 95% accurate" without asking "how confident are we in that 95%?" or "does our uncertainty estimate actually work?" This gap between aerospace's rigorous uncertainty quantification and ML's often hand-wavy confidence estimates became the core question driving this project.

## Research Question

**How can we reliably quantify uncertainty in machine learning classifier performance metrics using methodologies inspired by aerospace engineering?**

The problem is straightforward: when deploying ML models in production, we need to know not just "the model is 95% accurate" but "we're 95% confident the accuracy is between 92-98%." More importantly, we need to _validate_ that our uncertainty estimates actually work—that when we claim 95% confidence, the true value really falls within our interval 95% of the time.

This project applies aerospace-style statistical rigor to ML: validating coverage empirically, comparing methods systematically, and testing whether these techniques scale from small classifiers to larger models. The goal is to bring the same level of statistical validation that aerospace engineering demands to machine learning deployment decisions.

## Approach

I'm systematically comparing six uncertainty quantification methods, drawing from both traditional statistics and aerospace engineering:

1. **Binomial model** (pass/fail tests → accuracy)
2. **CLT-based confidence intervals** for accuracy
3. **Error propagation (Delta/RSS)** for F1 score
4. **Reliability/life-testing view** (failure rate)
5. **Monte Carlo/bootstrap** (simulation-based)
6. **FEA-style error energy metric** (energy norm of error)

The key is validation: I'm not just implementing these methods, but empirically testing whether they actually work. For each method, I'm checking:

- **Coverage**: Do their 95% intervals actually contain the "true" metric 95% of the time? This is the critical validation step that many papers skip.
- **Interval width**: Are the intervals tight or wide? Tighter is better, but only if coverage is correct.
- **Computational cost**: How long do they take to compute?
- **Behavior**: Do they behave sensibly with sample size, class imbalance, etc.?

## Future Direction

This foundational work establishes validated methods on small-scale classifiers. The natural next step is testing whether these uncertainty quantification techniques scale to large language models, vision transformers, and production-scale deployment scenarios. The ultimate goal is developing uncertainty-aware model selection and deployment protocols that prevent overconfident models from being deployed—especially critical in safety-sensitive applications where overconfidence can be dangerous.

## Setup and Tech Stack (Pipenv + sklearn)

### Create project and environment

```bash
pipenv shell # activate virtual environment
jupyter notebook # or jupyter lab
```

### Project Structure

```text
ai_aero_uncertainty/
  Pipfile
  Pipfile.lock
  data/                      # (optional, if you download a dataset)
  notebooks/
    01_data_and_model.ipynb        # load data, train model
    02_uncertainty_methods.ipynb   # compute intervals per block
    03_results_and_plots.ipynb     # compare methods
  src/
    data_utils.py
    model_utils.py
    metrics_uncertainty.py        # binomial, CLT, delta-F1, reliability, bootstrap
    fea_energy.py                 # FEA-style error energy metric
  README.md
```

## Choose a Dataset and Train One Simple Model

### Pick a dataset

We want a clean, standard classification problem so the focus is on uncertainty, not on "fixing" the model.

Use a built-in sklearn dataset first:

- `load_breast_cancer()` (binary, small but fine for a first pass), or
- `load_digits()` (multiclass)

> Later we can switch to a bigger Kaggle dataset for more sample size

### Dataset split into Train / Reference / Pool Split

We want to give our data different jobs:

1. Train set - used only to fit the model (train set is where the model "learns")
2. Reference set - used to estimate the model's "true" performance (Ref set is where we ask "how good is the model, what's it real accuracy?")
3. Pool set - used to draw multiple random samples to test uncertainty methods (This is a playground to see how well our uncertainty methods work)

We implement this split in `01_data_and_model.ipynb`.
