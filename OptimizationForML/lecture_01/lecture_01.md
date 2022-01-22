# Convexity

## Theory of Convex Functions

### Mathematical Background

#### Notation

Let $\mathbf{u}$ a vectors in $\mathbb{R}^d$ (we use bold font), and for their
coordinates $ \mathbf{u} = (u_1, \dots , u_d) ^ {\top} $. 
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
    0 \leq \|\lambda \mathbf{u} + \mathbf{v}\|, \qquad 
    \forall \lambda \in \mathbb{R}. 
$$
Further, 
\begin{equation*}
    \begin{aligned}
        \|\lambda \mathbf{u} + \mathbf{v}\| ^ 2 
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
of calculus see e.g. {cite:ts}`Spivak2008` for details.

#### The mean value theorem
Before enunciate the mean value Theorem we write a previous 
result that gives a tool to prove the mean value Theorem quite
clear and easy.
```{prf:theorem} Rolle's Theorem
   Let $f$ a realvalued function 
   $
    f: [a,b] \subset M \to \mathbb{R}
   $.
   If $f$ is continuous on $[a,b]$ and differentiable on
   $(a, b)$, then there exist $x \in (a,b)$ such that
   $f'(x) = 0$.
```

```{prf:theorem} Mean value Theorem
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
See for example {cite:ts}`Marsden1993`.

```{prf:definition}
:label: dfn_differentiability
Let $f: \mathrm{dom}(f)\subseteq \mathbb{R}^d \to \mathbb{R}^{m}$.
A function $f$ is called differentiable at 
$\mathbf{x}\subseteq \mathrm{dom}(f)$ if there exist an 
$(m\times d)-$ matrix $A$ an error function 
$r:\mathbb{R}^{d}\to \mathbb{R}$ defined on a neighborhood of
the vector $\mathbf{0} \in \mathbb{R}^d$ such that for all
$\mathbf{y}$ in a neighborhood of $\mathbf{x}$,

\begin{equation*}
    \begin{aligned}
        & f(\mathbf{y}) =
        f(\mathbf{x}) + A (\mathbf{y}-\mathbf{x}) + 
        r(\mathbf{y}-\mathbf{x}),
        \\
        \text{ such that } &
        \\
        & \lim_{\mathbf{v} \to \mathbf{0}}
            \frac{
                \|r(\mathbf{v})\|
            }{
                \|\mathbf{v}\|
            }
            =0 .
    \end{aligned}
\end{equation*}
```
Results that matrix $A$ is unique. This matrix is the so-called 
differential or Jacobian of $f$ at $\mathbf{x}$. 
We will denote it by $Df(\mathbf{x})$. More precisely, 
$Df(\mathbf{x})$ is the matrix of partial derivatives at
the point $\mathbf{x}$,

$$
    \begin{aligned}
         D f ( \mathbf{x} )_{ij} 
        &= 
            \frac{\partial f_i}{\partial x_j}(\mathbf{x})
        \\
        & = 
        \begin{pmatrix}
            \frac{\partial f_1}{\partial x_1}(\mathbf{x})
                & \cdots
                & \frac{\partial f_d}{\partial x_1}(\mathbf{x})
            \\
                \vdots & \ddots & \vdots
            \\ 
            \frac{\partial f_1}{\partial x_m}(\mathbf{x})
                & \cdots
                & \frac{\partial f_d}{\partial x_m}(\mathbf{x})
        \end{pmatrix}.
    \end{aligned}
$$(eqn_jacobian)

We say that  $f$ is differentiable  if for all 
$x \in \mathbf{dom}(f)$ the above limit exists
(which implies that $\mathrm{dom}(f)$ is open).

Differentiability at $\mathbf{x}$ means that in some neighborhood of 
$\mathbf{x}$, $f$ is approximated by a (unique) affine function 
$
    f(\mathbf{x})
    + D f(\mathbf{\mathbf{x}})(\mathbf{y}
    − \mathbf{x})
$, up to a sublinear error term. If $m = 1$, 
$Df(\mathbf{x})$ is a row vector typically denoted by 
$\nabla f(\mathbf{x})$, where the (column) vector 
$\nabla  f(\mathbf{x})$ is called the gradient of
$f$ at $\mathbf{x}$. Geometrically, this means that the graph of
the affine function 
$
    f(\mathbf{x}) 
    + \nabla f(\mathbf{x}) ^ {\top}
    (\mathbf{y} − \mathbf{x})
$ is a tangent hyperplane to the graph of $f$ 
at $(\mathbf{x}, f(\mathbf{x}))$.

```{prf:lemma} Chain rule
Let 
$
    f: \mathrm{dom}(f) \to \mathbb{R}^m,
     \mathrm{dom}(f) \subseteq \mathbb{R}^d$
and $\mathbb{g}: \mathrm{dom}(g) \to \mathbb{R}^d$. 
Suppose that $g$ is differentiable at $\mathbf{x} \in \mathrm{dom}(g)$ 
and that $f$ is differentiable at $g(x) \in \mathrm{dom}(f)$. 
Then $f \circ g$ (the composition of $f$ and $g$) is
differentiable at $\mathbf{x}$, with the differential given by

$$
    D(f \circ  g)(x) = Df(g(x))\ Dg(x).
$$
```
The following change of variable is very useful.
Let $f : \mathrm{dom} (f) \to \mathbb{R}^m$ be a differentiable
function with (open) convex domain, and fix 
$\mathbf{x}, \mathbf{y} \in \mathrm{dom}(f)$. There is an open 
interval $I$ containing $[0, 1]$ such that
$
    \mathbf{x} + t (\mathbf{y} − \mathbf{x}) \in \mathrm{dom}(f)
$ for all $t \in I$. 
Define $g : I \to \mathbb{R} ^ d$ by 

$$
    g(t) := \mathbf{x} +  t (\mathbf{y} - \mathbf{x})
$$ (eqn_convex_changue)

and set 

$$
    h := f \circ g 
        = f(\mathbf{x} + t (\mathbf{y} - \mathbf{x})).
$$  

Then $ h:[0,1] \to \mathbb{R}^{m}$ and 

$$
    \begin{aligned}
        h^\prime (t) = D f(g(t))\ D g(t)
            = D f ( \mathbf{x} + t (\mathbf{y} - \mathbf{x}))\  
                (\mathbf{y} - \mathbf{x})
    \end{aligned}
$$

### Convex sets
```{prf:definition}
A set $C \subseteq \mathbb{R}^{d}$ is convex if for any two
points $\mathbf{x}, \mathbf{y} \in C$, the connecting line
segment is contained in $C$. That is

$$
    \lambda \mathbf{x} (1 - \lambda) \mathbf{y} \in C, 
    \quad \forall \lambda \in [0, 1].
$$
```

```{prf:proposition}
Let $X_1$ and $X_2$ two convex sets from $\mathbb{R}^d$. 
Then 
$ X_1 \cap X_2$ is a convex set.
```
#### The convex hull 
We need the following lineal combination kinds.
```{prf:definition}
Let 
$
    M:= \{ \mathbf{x}_1,\dots, \mathbf{x}_m\}
    \subset
    \mathbb{R} ^ d  
$, take
        
$$
    \mathbf{x} := \sum_{i=1}^ {m} \alpha_i \mathbf{x}_i.
$$
Then we say that $\mathbf{x}$ is
- lineal combination of $M$ if $\alpha_i \in \mathbb{R}$.
- non-negative-lineal combination of $M$ if  $\alpha_i > 0$.
- convex-lineal combination if $\alpha_i \in [0, 1]$ and
  $ \displaystyle \sum_{i=1} ^ m \alpha_i = 1$.
```
    
```{prf:definition} Convex hull
Let $S$ and arbitrary set of $\mathbb{R}^d$. We denote the convex
hull of $S$ by
    
$$
    H(S) := 
        \left \{
            \mathbf{x} \in S:
            \mathbf{x} = 
            \sum_{i=1} ^ d
                \alpha_i \mathbf{x}_i,
                \text{ is lineal-convex-combination of elments from $S$}
        \right \} .
$$
```
#### Operations that preserve convexity
The following Propositions establish the convexity for the image of linear
functions and the set that results of common set operations.
```{prf:proposition}
Let $S \subset \mathbb{R} ^ d$ a convex set and 
$f: \mathbb{R}^d \to \mathbb{R} ^ m$. Then the image of 
$S$ under $f$, $f(S)$ is a convex set. 
```
```{prf:proof} **[prj_q_01]. **
    Let $\mathbf{x} ^ {\prime}, \mathbf{y} ^ {\prime} \in f(S)$
    and $\lambda \in [0, 1]$.
```
```{prf:proposition}
Let $\left\{X_i \subset \mathbb{R} ^ d, \  i \in I  \right\}$ 
a family of convex sets. Then the following sets also are convex.
1.  The arbritrary intersection
    $
        X:= \displaystyle \bigcap \limits_{i \in I} X_i
    $
2. The cartesian product
    $X:= \displaystyle \prod_{i \in I} X_i$

3. If $I:= \{1, \dots, m \}$ the finite sum
    $
        X := \sum_{i = 1} ^ {m}
            X_i 
    $, where 
    
    $$
         \sum_{i = 1} ^ {m}
            X_i :=
                 \left\{
                    \mathbf{x} \in \mathbb{R}^d :
                    \mathbf{x} = 
                        \sum_{i = 1} ^ {m} 
                        \mathbf{x}_i,
                    \quad
                    \mathbf{x}_i \in X_i, \ i = 1, \dots, m
                 \right\} . 
    $$
```


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

```{bibliography}
:style: plain
:filter: docname in docnames
```
