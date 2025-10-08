from dash import html, dcc
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# Generate static intuition plots with Matplotlib
# ------------------------------------------------------------

os.makedirs("assets", exist_ok=True)

# Figure 1: Perfect fit with two points
x1 = np.array([1, 2])
y1 = np.array([2, 4])

plt.figure(figsize=(5, 4))
plt.plot(x1, y1, color="blue", label="Line of Best Fit")
plt.scatter(x1, y1, color="black", zorder=5)
plt.title("Perfect Fit on Two Points (SSE = 0)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("assets/fit_two_points.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 2: Same line, but more points added
# Figure 2: Same line, but more points added
x2 = np.linspace(0, 3, 7)             # cover a wider x range
y2 = np.array([0, 0.6, 1.4, 2.4, 3.0, 3.6, 4.2])  # purposely off the line
m, b = 2, 0                           # original line y = 2x

# continuous line for visualization
x_line = np.linspace(0, 3, 100)
y_line = m * x_line + b
y_pred = m * x2 + b

plt.figure(figsize=(5, 4))
plt.plot(x_line, y_line, color="blue", label="Original Line (y = 2x)", linewidth=2)
plt.scatter(x1, y1, color="red", label="Original Data", zorder=5, s=40)
plt.scatter(x2, y2, color="black", label="New Data", zorder=5, s=40)


# Residuals: red dotted vertical lines
for i in range(len(x2)):
    plt.plot([x2[i], x2[i]], [y_pred[i], y2[i]], color="red", linestyle="dotted", linewidth=1.5)

plt.title("Adding More Points Introduces Error (SSE > 0)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("assets/fit_more_points.png", dpi=300, bbox_inches="tight")
plt.close()


# ------------------------------------------------------------
# Layout for introduction section
# ------------------------------------------------------------

layout = html.Div([
    html.H2("Introduction"),

    dcc.Markdown(
        r"""
Linear regression is one of the basic concepts in statistics and data analysis. 
We are looking for a straightforward pattern in data so we can predict on thing from another.
We can take an outcome $y$ and try to explain it using predictors $x_1, x_2, \dots, x_p$.  
The model is written as:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon
$$

The coefficients are chosen by **Ordinary Least Squares (OLS)**, whose whole point is to find the line that minimizes the distance of our predictions are from the real data.

In other words, the OLS attempts to minimize the sum of squared errors (SSE):

$$
\text{SSE}(\beta) = \sum_{i=1}^n \big(y_i - \hat{y}_i\big)^2
$$

OLS works well when predictors are few and independent.  
But when predictors are **many**, **highly correlated**, or the model is **poorly trained**,  
it can lead to large, unstable coefficients and **overfitting**.
""",
        mathjax=True
    ),

    html.H2("Building Intuition with Simple Linear Regression"),

    html.P(
        "To build intuition, let’s imagine the case of simple linear regression (SLR) "
        "with the formula y = mx + b. Suppose we train the model using only two data points."
    ),

    html.Img(src="/assets/fit_two_points.png",
             style={"width": "60%", "display": "block", "margin": "auto"}),

    html.P(
        "With only two points, the fitted line passes exactly through both. "
        "If we calculate SSE we get zero because the model fits perfectly."
    ),

    html.P(
        "Now, imagine we add more data points to the same plot but keep the same fitted line."
    ),

    html.Img(src="/assets/fit_more_points.png",
             style={"width": "60%", "display": "block", "margin": "auto"}),

    html.P(
        "Now the model no longer fits perfectly the added points deviate from the line, creating residuals. "
        "The sum of squared errors (SSE) grows because the model’s predictions no longer match all data points exactly. "
        "This simple setup illustrates a key problem:"
    ),

    html.H3(
        "overfitting.",
        style={
            "textAlign": "center",
        }
    ),

    html.P(
        "our model fits the small training sample too precisely "
        "and fails to generalize when new data are added"
    ),

    html.P("This is just one example of overfitting. Others include:"),

    html.Ul([
        html.Li("Too many parameters for the amount of data"),
        html.Li("Highly correlated predictors"),
        html.Li("Insufficient regularization or noise handling"),
        html.Li("Model trained too long or without validation"),
    ]),

    html.P(
        "Regularized regression helps correct this by adding a penalty term to the OLS objective function, "
        "which discourages overly complex models. In practice, this is done by adding one of two penalties: "
        "L2 (Ridge) or L1 (Lasso) regularization."
    ),
])
