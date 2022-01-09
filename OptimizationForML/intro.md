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
```{note} The final grade is the sum of the acommulated points.
You can always get as many points as you can bue no more than 
100. Taht is, if you make 60 points with Exams and projects 
but not with activites or exposition then you can recore point
by a recovering project. If you obtain the 80 % of the points
of the lectures activities the you can exempt the writing exam
corresponding to the module course . 


|   **Delivery**    |   **Points**       |     **Deadline**     |
|:-----------------:|:------------------:|:--------------------:|
|   Final Project   |        30          | May 13 2022 at 23:59 |
| Writing Exams (3) | 15 (5 per exam)    |                      |
| Code Exams   (3)  | 15 (5 per exam)    |                      |
|    Exposition     | 15 (5 per session) |                      |
| Lecture activity  | 1 per activity     |     Each Friday      |

# Recovering projects
```{important}
Agree with teacher topic and delivery time.
10 point per project.
```
 
# Bibliography
```{bibliography}
:style: plain
:filter: docname in docnames
```
