# TSA
My built-from-scratch derivate-based hyperplane fitter and its implementation in predicting stock prices

## Derivative-based hypeplane fitter
Lets consider the input matrix as 
$$
A = \begin{bmatrix}
    a_{11} & a_{12} & \cdots & a_{1n} \\
    a_{21} & a_{22} & \cdots & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$
To fit an intercept A becomes
$$
A = \begin{bmatrix}
    a_{11} & a_{12} & \cdots & a_{1n-1} & 1\\
    a_{21} & a_{22} & \cdots & a_{2n-1} & 1\\
    \vdots & \vdots & \ddots & \vdots & 1\\
    a_{m1} & a_{m2} & \cdots & a_{mn-1} & 1
\end{bmatrix}
$$

