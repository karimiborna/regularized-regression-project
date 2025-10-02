from dash import html, dcc

layout = html.Div([
    html.H2("Introduction"),
    dcc.Markdown(
        r"""
Linear regression is a fundamental model for predicting an outcome $y$ based on predictors $x_1, x_2, \dots, x_p$.  
The model is written as:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon
$$

Here:
- $y$ is the response variable,
- $x_j$ are the predictors,
- $\beta_j$ are the coefficients,
- $\epsilon$ is the error term.

The coefficients are chosen by **Ordinary Least Squares (OLS)**, which minimizes the sum of squared errors (SSE):

$$
\text{SSE}(\beta) = \sum_{i=1}^n \big(y_i - \hat{y}_i\big)^2
$$

OLS works well when predictors are few and independent. But when predictors are **many** or **highly correlated**, OLS can lead to:
- Large, unstable coefficients  
- Overfitting the training data  
- Poor generalization to new data  

To address this, we use **regularization**, which modifies the OLS loss by adding a penalty term on the size of the coefficients.
""",
        mathjax=True  
    )
])
