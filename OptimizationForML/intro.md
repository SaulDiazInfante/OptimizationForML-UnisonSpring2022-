# Welcome to the Optimization for Machine Learning Course

This is the first version for a series of 12 Lectures about 
numerical optimization methods that have direct applications in
Machine Learning. The material course is based on the
Martin Jaggi and Nicolas Flammarion Course at EPFL,
the book of {cite:ts}`VanderPlas2016` and other sources.

```{admonition} Optimization for Machine Learning CS-439
[Playlist on youtube](https://www.youtube.com/playlist?list=PL4O4bXkI-fAeYrsBqTUYn2xMjJAqlFQzX)
```
## Motivation
The problem is finding a vector $x \in \mathbb{R}^d$ that 
optimize a given function   $f:\mathbb{R}^d \to \mathbb{R}$.
By convention, we will formulate this optimization problem as
minimization problem.
\begin{equation*}
    \begin{aligned}
        \text{minimize } & f(\mathbf{x}) \\
            \mathbf{x} \in \mathbb{R}^d
    \end{aligned}
\end{equation*}

```{admonition} Typically assumptions
- $f$ is continuous and differentiable
- $\mathrm{dom}{f}$ usually would be a convex set
```

```{important} Applications almost everywhere.
    In this course we consdier applications from
    data sicence (machine learning), control theory and  
    other data analytics. We particullarly focus on
- Mathematical Modeling
    - defining and measuring the machine learning model
- Computational Optimization
    - learning the model parameters
- Theory vs. practice. There exist verry used 
libraries for example, according to the 
program language
    - Python
        - SciPy
        - Scikit-learn
        - Theano
        - TensorFlow
        - Keras
        - PyTorch
    - R
        - CARAT
        - Random Forest
        - E1071
        - RPart
    - Julia
        - Mocha
        - Knet
        - SciKitLearn.jl
        - Flux
        - MLBase.jl: 
        - Strada
```
Here we put stress on some python packages and
will review the intuitions for computational
convex optimization methods. We will focus in descend methods
like
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Coordinate Descent

# Prerequisites
- Linear algebra, calculus, mathematica analysis.
- Manage data structures as list, arrays from any 
programming language.
- Numerical methods and numerical analysis
# Evaluation
    Final Project 60 % of the grade.
    Deadline May 13 2022 at 23:59
# Recovering projects
# Bibliography
```{bibliography}
:style: plain
:filter: docname in docnames
```
