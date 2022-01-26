# <center>Project 3 on Mathematics in AI</center>

Subject: Square Root

Name: Hesam Mousavi

Student number: 9931155

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true},
jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
TeX: {
extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
equationNumbers: {
autoNumber: "AMS"
}
}
});
</script>

## First Idea

### Using eigenvalue and eigenvector

Let's find the eigenvalue and eigenvector of $\sqrt{A}$

Just as with the real numbers, a real matrix may fail to have a real square root, but
have a square root with **complex-valued** entries. Some matrices have **no square root**.

### Positive semidefinite(P.S.D) matrices

When matrix M is P.S.D? when M is **symmetric** matrix and
$\forall x \rightarrow x^{\top} M x \geqslant 0$ then call M as P.S.D matrix

A square real matrix is positive semidefinite if and only if $A=B^{\top} B$ for some matrix B. There can be many different such matrices B. A positive semidefinite matrix A can also have many matrices B such that $A=B B$. However, A always has precisely one square root B(**principal**, **non-negative**, or **positive square root**) that is positive semidefinite (and hence symmetric).

The principal square root of a real positive semidefinite matrix is real. The principal square root of a positive definite matrix is positive definite; more generally, the **rank** of the principal square root of A is the same as the rank of A.

An n×n matrix with n distinct nonzero eigenvalues has $2^{n}$ square roots. Such a matrix, A, has an **eigendecomposition** $V D^{1 / 2} V^{-1}$ where V is the matrix whose columns are eigenvectors of A and D is the diagonal matrix whose diagonal elements are the corresponding n eigenvalues $\lambda_{i}$. Thus the **square roots** of A are given by $V D^{1 / 2} V^{-1}$, where $D^{1 / 2}$ is any square root matrix of D, which, for distinct eigenvalues, must be diagonal with diagonal elements equal to square roots of the diagonal elements of D; since there are two possible choices for a square root of each diagonal element of D, there are $2^{n}$ choices for the matrix $D^{1 / 2}$.(notes: 1- $D$ may have complex values. 2- $A$ must has $n$ linearly independent eigenvectors)

So $\sqrt{A} = A^{1 / 2} = V D^{1 / 2} V^{-1}$ because $\left(V D^{\frac{1}{2}} V^{-1}\right)^{2}=V D^{\frac{1}{2}}\left(V^{-1} V\right) D^{\frac{1}{2}} V^{-1}=V D V^{-1}=A$

## Observations

With the eigendecomposition method, I found matrix $A$ and also $\forall i: \sum_{j=1}^{N} a_{i, j}=1$ in all 100 random matrices
