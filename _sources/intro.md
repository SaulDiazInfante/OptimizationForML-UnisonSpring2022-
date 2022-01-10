# Welcome to the Optimization for Machine Learning Course

This attempt is the first version of 12 Lectures about 
numerical optimization methods that have direct applications
on the Martin Jaggi and Nicolas Flammarion Course at EPFL,
the book of VanderPlas {cite:ts}`VanderPlas2016` and other sources. 

```{admonition} Optimization for Machine Learning CS-439
[Playlist on youtube](https://www.youtube.com/playlist?list=PL4O4bXkI-fAeYrsBqTUYn2xMjJAqlFQzX)
```
## Motivation
Our main problem is finding a vector $x \in \mathbb{R}^d$ 
that optimizes a given function
$f:\mathbb{R}^d \to \mathbb{R}$.
By convention, we will formulate this optimization problem 
as a minimization problem.

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

## Theory vs. practice. 
There exist a bunch of very used 
libraries. For example, according to python, R, and Julia,
the following are popular.

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
will review the formal intuitions of computational
convex optimization methods. We will focus on descending methods like
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Coordinate Descent

# Prerequisites
- Linear algebra, calculus, mathematical analysis.
- Manage data structures like lists, arrays from any 
programming language (C, C++, Fortran, Matlab, and similar).
- Numerical methods and numerical analysis

# Evaluation
```{note} The final grade is the sum of the accumulated points.
You can always get as many points as possible but no more than 100.
That is, if you make 60 points with Exams and Projects but not with 
Activities or Exposition,
you can recuperate points by a recovering project. 
``` 

```{important}
If you rebase 80 %  of the points of the lectures activities,
you can exempt the writing exam corresponding to the module.
```
 


|   **Delivery**    |     **Points**     |                      |
|:-----------------:|:------------------:|:--------------------:|
|   Final Project   |         30         | May 13 2022 at 23:59 |
| Writing Exams (3) |  30 (10 per exam)  |      **Dates**       |
 |                   |                    |   Jan 28 2022 at *   |
|                   |                    |   Feb 25 2022 at *   |  
|                   |                    |   Mar 25 2022 at *   |
 | Code Exams   (3)  |  30 (10 per exam)  |    **Deadlines**     |
 |                   |                    | Feb 5 2022 at 23:59  |
|                   |                    | Mar 5 2022 at 23:59  |
|                   |                    | Apr 02 2022 at 23:59 |
|  Exposition (2)   | 10 (5 per session) |                      |
| Lecture activity  |   1 per activity   |     Each Friday      |

# Recovering projects
```{important}
Agree with teacher topic and delivery time.
10 point per project.
```
 
# Bibliography
The main references for the thoery are the Books of
{cite:ts}`Boyd2004,Guller2010`. The applications to Data Science and other
programming topics can be found on {cite:ts}`VanderPlas2016`.

```{bibliography}
:style: plain
:filter: docname in docnames
```
