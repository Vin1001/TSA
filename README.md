# TSA

My built-from-scratch derivative-based hyperplane fitter and its implementation in predicting stock prices

## Derivative-based hypeplane fitter
Let us consider the input matrix as

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
    a_{11} & a_{12} & \cdots & a_{1n} & 1 \\
    a_{21} & a_{22} & \cdots & a_{2n} & 1 \\
    \vdots & \vdots & \ddots & \vdots & \vdots \\
    a_{m1} & a_{m2} & \cdots & a_{mn} & 1
\end{bmatrix}
$$

Ordinary Least Squares(OLS) estimator for the input matrix is defined to be

$$
\beta = (A^T A)^{-1} A^T y
$$

where y is the target vector and β is the weight vector. However, OLS is highly sensitive to outliers since squaring induces unnecessary bias in the model. To overcome the problem of squaring, I developed derivative-based estimation, which uses:

$$
\beta = (D^T A)^{-1} D^T y
$$

where 

$$
D = \begin{bmatrix}
    1/a_{11} & 1/a_{12} & \cdots & 1/a_{1n} & 1 \\
    1/a_{21} & 1/a_{22} & \cdots & 1/a_{2n} & 1 \\
    \vdots & \vdots & \ddots & \vdots & \vdots \\
    1/a_{m1} & 1/a_{m2} & \cdots & 1/a_{mn} & 1
\end{bmatrix}, (a_{ij} \neq 0\)
$$

### Intuitive meaning:
#### - Through this approach the change of an input variable with respect to other input variables is found out.
#### - The target variable per unit input variable also conveys an intuitive meaning as to which variable has the most impact on the target vector.
#### - Reduced sesitivity to outliers- Using the model with z-score normalized data, will automatically diminish the effect of values which have more deviation from mean.

## Stock Price Application

Using a sliding window application for the derivative-based fitter, not only have i developed a sliding window stock price predictor but also a prediction mechanism to incorporate the fluctuations in the stock prices using binary trees. The figure below is an illustration as to how binary trees have applied in prediction and accomodate volatility:

![notebooks\binary_illustration.png](https://github.com/Vin1001/TSA/blob/main/notebooks/binary_illustration.png)

Here RF is called the Risk Factor ranging between [0, 1]. RF at tick = t is calculated as the ratio of difference between prediction for t and actual value at t, to twice of nth percentile absolute error(for eg. 80th percentile error if considered significant).
#### For demonstration of the model check out notebooks directory

### License
This project is licensed under the Apache 2.0 License.

### Performance
Here's a comparison of the proposed model's performance with other time series forecasting model:
Google's AutoBNN
![performance/AutoBNN.png](https://github.com/Vin1001/TSA/blob/main/performance/AutoBNN.png)

Amazon's Chronos
![performance/chronos_comparison.png](https://github.com/Vin1001/TSA/blob/main/performance/chronos_comparison.png)


### Contact
For questions or feedback, feel free to reach out to vinayaksoni704@gmail.com


Disclaimer:
The projects and analyses provided in this repository are for educational purposes only and should not be construed as financial advice.
