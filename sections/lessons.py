from dash import html, dcc
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("assets", exist_ok=True)

# Original data and line
x2 = np.linspace(0, 3, 7)
y2 = np.array([0, 0.6, 1.4, 2.4, 3.0, 3.6, 4.2])
m1, b1 = 2, 0      # original OLS line
m2, b2 = 1.3, 0.1  # shrunk Ridge-style line

x_line = np.linspace(0, 3, 100)
y_line1 = m1 * x_line + b1
y_line2 = m2 * x_line + b2

# Predictions from each model at data x positions
y_pred_ols = m1 * x2 + b1
y_pred_ridge = m2 * x2 + b2

plt.figure(figsize=(5, 4))

# Scatter data
plt.scatter(x2, y2, color="black", label="Data", zorder=5, s=40)

# Original OLS fit (blue)
plt.plot(x_line, y_line1, color="blue", label="Original Fit (y = 2x)", linewidth=2)

# Ridge fit (green, dashed)
plt.plot(x_line, y_line2, color="green", linestyle="--",
         label="Ridge Fit (y = 1.3x + 0.1)", linewidth=2)

# Residuals: red dotted lines from each point to Ridge line
for i in range(len(x2)):
    plt.plot([x2[i], x2[i]], [y2[i], y_pred_ridge[i]],
             color="red", linestyle="dotted", linewidth=1.5)

plt.title("Ridge Regression Shrinks the Coefficient (m) Toward Zero")
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
        "New Loss = New SSE + penalty = 1 + 1.3^2 = 2.69.",
        style={
            "textAlign": "center",
        }
    ),

    html.P(
        "So by doing this, we can see that minimizing the penalty term, "
        "we have shrunk the coefficients and found a better fitting line."
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
