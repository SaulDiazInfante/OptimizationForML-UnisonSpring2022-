# Convexity

## Theory of Convex Functions

### Mathematical Background

#### Notation

Let $\mathbf{u}$ a vectors in $\mathbb{R}^d$ (we use bold font), and for their
coordinates $ \mathbf{u} = (u_1, \dots , u_d)$. 
We consider vectors as column vectors, 
unless they are explicitly transposed. So 
$\mathbf{u}$ is a column vector, and its transpose $\mathbf{u}^\top$, 
is a row vector. We denote by $\mathbf{u}^{\top} \mathbf{v}$ 
the usual scalar product, that is

$$ 
    \mathbf{u}^{\top} \mathbf{v} = \sum_{i = 1} ^ d u_i v_i. 
$$ 

We use
$\|\cdot\|$ to denote the euclidean norm in $\mathbb{R}^d$. Thus

$$ \|\mathbf{u}\|^ 2 = \mathbf{u} ^ {\top} \mathbf{u} = \sum_{i=1} ^ d x_i^2. $$

#### The Cauchy-Schwarz inequality

```{prf:lemma} Cauchy-Schwarz inequality
:label: thmCauchuSchwarz 
 
See e.g. {cite:t}`Marsden1993`. 
For all $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ , 

$$
    |\mathbf{u}^{T} \mathbf{v}| \leq  
    \| \mathbf{u}\| \| \mathbf{v} \|.
$$
```
```{prf:proof}
    Let $\mathbf{u},  \mathbf{v}$ in $\mathbb{R}^d$. Note that
    
$$
    \leq 0|\lambda \mathbf{u} + \mathbf{v}|, \qquad 
    \forall \lambda \in \mathbb{R}. 
$$
Further, 
\begin{equation*}
    \begin{aligned}
        |\lambda \mathbf{u} + \mathbf{v}| ^ 2 
        &=
            (\lambda \mathbf{u} + \mathbf{v})^{\top}
            (\lambda \mathbf{u} + \mathbf{v})
        \\
        &=
       \underbrace{
            \lambda ^ 2  \mathbf{u}^{\top} \mathbf{u}
            + 2 \lambda  \mathbf{u}^{\top} \mathbf{v}
            + \mathbf{v}^{\top} \mathbf{v}
       }_{:= p(\lambda)}
    \end{aligned}
\end{equation*}
Since the square polinomial $p(\lambda)$ reachs its minimum at

$$
    \lambda = 
        - \frac{\mathbf{u}^{\top} \mathbf{v}}{
            \| \mathbf{u} \| ^ 2
        },
$$
and $p(\lambda) \geq 0$, we deduce that
\begin{equation*}
    \begin{aligned}
        &
        p(\lambda) = 
            \left(
                - \frac{\mathbf{u}^{\top} \mathbf{v}}{
                        \| \mathbf{u} \| ^ 2}
            \right)
            \mathbf{u}^{\top} \mathbf{u}
            +
            2 \left(
                - \frac{\mathbf{u}^{\top} \mathbf{v}}{
            \| \mathbf{u} \| ^ 2}
            \right)
            \mathbf{u}^{\top} \mathbf{v}
            +
            \mathbf{u}^{\top} \mathbf{v}
        \\
        \text{that is} &
        \\
        &
        \|\mathbf{v}\|^2
        - \frac{\mathbf{u}^{\top} \mathbf{v}}{\| \mathbf{u} \| ^ 2}
        \geq 0.   
    \end{aligned}
\end{equation*}    

Then 

$$    
        \left(
            \mathbf{u}^{\top} \mathbf{v}
        \right)^ 2 
        \leq \|\mathbf{u}\| ^ 2 \|\mathbf{u}\|^2.
$$    
Therefore, taking square root we obtain the regarding
inequality \qued .
```

#### The spectral norm

```{prf:definition} Spectral norm
Let $A$ be an $(m \times d)$-matrix. Then

$$
    \| A\|:= 
        \max_{\mathbf{v} \in \mathbb{R}^d, \mathbf{v} \neq 0} 
                \frac{\|A \mathbf{v}\|}{\|\mathbf{v}\|}
        =
        \max_{\mathbf{v} \in \mathbb{R}^d, \mathbf{v} = 1}
            \|A \mathbf{v}\|.    
$$
```
We also recall two very important results from
classic calculus: the mean value theorem and the fundamental Theorem
of calculus see e.g. [Cite Spivak ] for details.
#### The mean value theorem
```{prf:theorem} Mean value theorem
Let $a < b$ be real numbers, and let 
$h :[a, b] \to \mathbb{R}$ be a continuous function
that is differentiable on $(a, b)$.    
Then there exists $c \in (a, b)$ such that

$$
    h'(c) = \frac{ h(b) − h(a)}{b − a}.
$$
```

#### The fundamental theorem of calculus

```{prf:theorem} Fundamental theorem of calculus
Let $a < b$ be real numbers, and let 
$h : \mathrm{dom} (h) \to \mathbb{R}$ be a differentiable 
function on an open domain 
$\mathrm{dom} (h)$  , and such that $h'$ is 
continuous on $[a, b]$. Then

$$
    h(b)-h(a) = \int_{a}^b 
        h'(t) dt.
$$
```
#### Differentiability

### Convex sets

#### The mean value inequality

### Convex functions

#### First-order characterization of convexity

#### Second-order characterization of convexity

#### Operations that preserve convexity

#### Minimizing convex functions

#### Strictly convex functions

#### Example: Least squares

#### Constrained Minimization

#### Existence of a minimizer

#### Sublevel sets and the Weierstrass Theorem

#### Examples

#### Handwritten digit recognition

#### Master’s Admission 
