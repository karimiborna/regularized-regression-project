from dash import html, dcc
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("assets", exist_ok=True)

# Original data and line
x1 = np.array([1, 2])
y1 = np.array([2, 4])
x2 = np.linspace(0, 3, 5) 
y2 = np.array([0, 0.6, 2.4, 3.6, 4.2])
m1, b1 = 2, 0      # original OLS line s
m2, b2 = 1.5, 0.1  # shrunk Ridge-style line

x_line = np.linspace(0, 3, 100)
y_line1 = m1 * x_line + b1
y_line2 = m2 * x_line + b2

# Predictions from each model at data x positions
y_pred_ols = m1 * x2 + b1
y_pred_ridge = m2 * x2 + b2
y_pred_ridge_x1 = m2 * x1 + b2

plt.figure(figsize=(5, 4))

# Scatter data
plt.scatter(x2, y2, color="black", label="New Data", zorder=5, s=40)
plt.scatter(x1, y1, color="red", label="Original Data", zorder=5, s=40)

# Original OLS fit (blue)
plt.plot(x_line, y_line1, color="blue", label="Original Fit (y = 2x)", linewidth=2)

# Ridge fit (green, dashed)
plt.plot(x_line, y_line2, color="green", linestyle="--",
         label="Ridge Fit (y = 1.5x + 0.1)", linewidth=2)

# Residuals: red dotted lines from each point to Ridge line
for i in range(len(x2)):
    plt.plot([x2[i], x2[i]], [y2[i], y_pred_ridge[i]],
             color="red", linestyle="dotted", linewidth=1.5)
    
for i in range(len(x1)):
    plt.plot([x1[i], x1[i]], [y1[i], y_pred_ridge_x1[i]],
             color="red", linestyle="dotted", linewidth=1.5)

plt.title("Ridge Regression Shrinks the Coefficients Toward Zero")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("assets/ridge_shrink.png", dpi=300, bbox_inches="tight")
plt.close()


layout = html.Div([
    html.H2("Ridge vs. Lasso, How do they fix this?"),
    dcc.Markdown(
        r"""

Ridge minimizes the **penalized SSE**, also known as the ** LOSS **:

$$
\mathcal{L}({\beta})
= \sum_{i=1}^{n} \big(y_i - \hat{y}_i\big)^2
+ \lambda \sum_{j=1}^{p} \beta_j^2
$$

- Focus on the term that comes at the end of the equation, lambda * the sum of the square of the slopes  
- This end term is the penalty added in by Ridge Regression, the **L2 penalty**  
- The penalty shrinks coefficients (Betas) toward zero.  
- Coefficients never become exactly zero.  
- Works well when predictors are correlated and all carry some signal.  
- **MINIMIZED LOSS IS THE IDEAL OUTCOME!**

""",
        mathjax=True
    ),

    html.P(
        "The ridge penalty will always work to shrink the coefficients to zero because of the setup of the equation. "
        "When you want to minimize an equation which is a sum, you want to minimize both parameters, "
        "in this case, both the OLS and the L2 penalty term."
    ),

    html.P("As you can see:"),

    html.Img(
        src="/assets/ridge_shrink.png",
        style={"width": "60%", "display": "block", "margin": "auto"}
    ),

    html.P(
        "Now we can compare: Originally:"
        "Now the new line: The new SSE + penalty = 1 + 1.3^2 = 2.69."
    ),

    html.H3(
        "Previous Loss = Previous SSE + penalty = 0 + 2^2 = 4. ",
        style={
            "textAlign": "center",
        }
    ),

    html.P(
        "Now the new line:"
    ),

    html.H3(
        "New Loss = New SSE + penalty = 1 + 1.5^2 = 3.25.",
        style={
            "textAlign": "center",
        }
    ),

    html.P(
        "So by doing this, we can see that minimizing the penalty term, "
        "we have shrunk the coefficients and found a better fitting line."
    ),

    html.H2(
        "Now that we have a stronger basic understanding of why ridge regression drives coefficients toward zero, lets get into the math"),

    dcc.Markdown(
        r"""

### Ridge Math

Remember the motivation: we have too many predictors. 

- Many predictors without penalization -> large prediction intervals and LS regression estimator may not uniquely exist.

Also: we have an "ill-conditioned" X
 
- Because the LS estimates depend on (X'X)^-1, we'd have problems computing the least squares estimate of B if X'X was singular or nearly singular
- The LS estimator of B may provide a good fit for training data, but it won't fit well to testing data

So, how do we get out of this pickle? One way to is abandon our need for an unbiased estimator.

We will require assuming only X's and Y have been centered, so we have no need for a constant term in the regression:

- X is a n by p matrix with centered columns
- Y is a centered n-vector

It was proposed the the LS estimator

$$
\hat{\beta} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}
$$


had instability which could be fixed by adding a constant lambda to the diagonals of the X'X matrix before taking its inverse.

The resulting matrix IS the ridge regression estimator

$$
\hat{\beta}_{\text{ridge}} = (\mathbf{X}'\mathbf{X} + \lambda \mathbf{I}_p)^{-1}\mathbf{X}'\mathbf{Y}
$$

The full constraint placed on the parameters (betas) to penalize the sum of squares is shown as: 

$$
\sum_{i=1}^{n} \left( y_i - \sum_{j=1}^{p} x_{ij}\beta_j \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
$$""",
    mathjax=True
),

    html.Img(
    src="/assets/geom_interp_ridge.png",
    style={"width": "70%", "display": "block", "margin": "auto"}
),

    html.H3("Geometric Interpretation of Ridge Regression"),

    html.P(
        "Another way to understand what Ridge Regression is doing is to look at it geometrically. "
        "The pink ellipses in this figure represent the contours of the residual sum of squares (RSS) "
        "each inner ellipse means a smaller RSS, and the very center is where OLS would be if WE DIDN'T CARE ABOUT COEFFICIENT SIZE!!! "
        "Unfortunately for those who wish this was simple, we do care, so we cannot accept this as our best answer. "
        "The blue circle shows the Ridge constraint: all possible coefficient combinations that satisfy the condition that the sum of their squared values stays below some constant threshold. "
    ),

    html.P(
        "The Ridge estimate happens where one of those RSS ellipses just touches the circle. "
        "That point represents the best balance between fitting the data well and keeping the coefficients small. "
        "If we only cared about minimizing RSS, we'd pick the center of the ellipse (OLS), but that usually gives us large, unstable coefficients. "
        "Using ridge finds us users a compromise of a slightly higher Residual Sum of Squares but much smaller coefficients. "
    ),

    html.P(
        "You can think of the penalty term as squeezing the circle smaller as λ increases. "
        "When λ = 0, the circle is huge and Ridge becomes the same as OLS. "
        "As λ grows larger, the circle shrinks, pulling the solution closer to the origin. "
        "In the extreme, as λ approaches infinity, all coefficients collapse toward zero. "
    ),


    dcc.Markdown(
        r"""
---

### Lasso Regression (L1 penalty)

Lasso minimizes:

$$
\mathcal{L}({\beta})
= \sum_{i=1}^{n} \big(y_i - \hat{y}_i\big)^2
+ \lambda \sum_{j=1}^{p} \lvert \beta_j \rvert
$$

- The L1 penalty encourages **sparsity**.  
- Some coefficients shrink all the way to zero (feature selection).  
- Useful when only a subset of predictors truly matter.  

---

### Key Lesson

- **Ridge**: smooth shrinkage — all coefficients decrease but remain in the model.  
- **Lasso**: selection — some coefficients drop to exactly zero.  

As \( \lambda \) increases, both add bias but reduce variance, improving generalization.  
""",
        mathjax=True
    )
])
