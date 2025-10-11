from dash import html, dcc
import matplotlib.pyplot as plt
import numpy as np
import os

import plotly.graph_objs as go
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.linspace(0, 10, 10).reshape(-1, 1)
slope_true = 1.5
intercept_true = 2
y = intercept_true + slope_true * X.flatten() + np.random.normal(0, 1, X.shape[0])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Generate lambdas only until slope equals 0 (or close)
lambdas = np.linspace(0, 8, 500)
coefs = []
lines = []
stopping_index = len(lambdas) - 1

for idx, alpha in enumerate(lambdas):
    lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    lasso.fit(X_scaled, y)
    slope_rescaled = lasso.coef_[0] / scaler.scale_[0]
    intercept_rescaled = lasso.intercept_ - lasso.coef_[0] * scaler.mean_[0] / scaler.scale_[0]
    coefs.append([slope_rescaled, intercept_rescaled])
    y_pred = intercept_rescaled + slope_rescaled * X.flatten()
    lines.append(y_pred)
    if np.abs(slope_rescaled) < 1e-5:  # near zero slope condition
        stopping_index = idx
        break

coefs = np.array(coefs)
lambdas = lambdas[:stopping_index + 1]
lines = lines[:stopping_index + 1]

data = [go.Scatter(
    x=X.flatten(), y=y, mode='markers', name='Data', marker=dict(color='red')
)]

steps = []
for i in range(len(lambdas)):
    slope = coefs[i, 0]
    steps.append(dict(
        method='update',
        args=[
            {'y': [y, lines[i]]},
            {'annotations': [
                dict(
                    x=5, y=max(y) + 2,
                    text=f"slope = {slope:.2f}",
                    showarrow=False,
                    font=dict(size=18, color='black')
                )
            ],
                'title': f'Lasso Regression (lambda={lambdas[i]:.2f})'}
        ],
        label=f'λ={lambdas[i]:.2f}'
    ))

data.append(go.Scatter(
    x=X.flatten(), y=lines[0], mode='lines',
    name='Lasso Regression Line', line=dict(color='orange')
))

layout_fig = go.Layout(
    title=f'Lasso Regression (lambda={lambdas[0]:.2f})',
    xaxis=dict(title='Weight'),
    yaxis=dict(title='Size'),
    sliders=[dict(
        active=0,
        steps=steps,
        currentvalue={"prefix": "lambda: "},
        pad={"t": 50},
    )],
    annotations=[dict(
        x=5, y=max(y) + 2,
        text=f"slope = {coefs[0, 0]:.2f}",
        showarrow=False,
        font=dict(size=18, color='black'))],
    showlegend=True)

lasso_fig = go.Figure(data=data, layout=layout_fig)

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

It was proposed the LS estimator

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
    style={"width": "50%", "display": "block", "margin": "auto"}
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

    $$
    \mathcal{L}({\beta})
    = \sum_{i=1}^{n} \big(y_i - \hat{y}_i\big)^2
    + \lambda \sum_{j=1}^{p} \lvert \beta_j \rvert
    $$

    Lasso Regression is very **similar** to Ridge Regression, but they have some very important **differences**.

    To understand those similarities and differences, let's look at this chart.
    """,
        mathjax=True
),

html.Img(
    src="/assets/lasso_001.jpg",
    style={"width": "50%", "display": "block", "margin": "auto"}),

         
dcc.Markdown(
    r"""
The chart show **Weight** and **Size** measurements from a bunch of mice.  
The **Red Dots** are **Training Data**, while the **Green Dots** are **Testing Data**.  

Then we fit a line to the **Training Data** using **Least Squares**, so we minimized the **sum of squared residuals**.  

From the chart, we can see that even though the line fit the **Training Data** very well (low **Bias**), it didn't fit the **Testing Data** very well (high **Variance**).
"""
),

html.Img(
    src="/assets/lasso_002.jpg",
    style={"width": "50%", "display": "block", "margin": "auto"}),

dcc.Markdown(
    r"""
Then we fit a line to the data using **Ridge Regression** to minimize:  
- **sum of squared residuals** + **λ** \* **the slope^2**  

So **Ridge Regression** is just **Least Squares** plus the **Ridge Regression Penalty**.  

The **Blue Ridge Regression** line did not fit the **Training Data** as well as the **Red Least Squares** line (more **Bias** less **Variance**).  

The idea was that by starting with a slightly worse fit, **Ridge Regression** provided better long term predictions.  

Now, let's focus on the **Ridge Regression Penalty**:  
- **λ** \* **the slope^2**  

If, instead of squaring the slope, we take the absolute value, then we have **Lasso Regression**!  
- **λ** \* **|the slope|**  

Just like Ridge Regression, **λ** can be any value from **0** to **positive infinity** and is determined using **Cross Validation**.  
"""
),

html.Img(
    src="/assets/lasso_003.jpg",
    style={"width": "50%", "display": "block", "margin": "auto"}),

dcc.Markdown(
    r"""
Like Ridge Regression, **Lasso Regression** (**Orange** line) results in a line with a little bit of **Bias**, but less **Variance** than **Least Squares**.  

Therefore, not only do Ridge Regression and Lasso Regression look **similar in expression**, but they also **do similar things**:  
They both make our predictions of **Size** less sensitive to this tiny Training Dataset.
"""
),

html.Img(
    src="/assets/lasso_005.jpg",
    style={"width": "50%", "display": "block", "margin": "auto"}),

dcc.Markdown(
    r"""
Both Ridge and Lasso Regression can be applied to complicated models that combine different types of data.  

In this case, we have two variables: **Weight** (continuous) and **High Fat Diet** (discrete).  
- **Size** = y-intercept + slope \* **Weight** + diet difference \* **High Fat Diet**  
- Sum of squared residuals + **λ** * (**|the slope|** + **|diet difference|**)  

Just like the Ridge Regression Penalty, **Lasso Regression Penalty** contains all the estimated parameters except for the y-intercept.  

But note that when Ridge and Lasso Regression shrink parameters, they don't have to shrink them all equally.
"""
),

html.Img(
    src="/assets/lasso_006.jpg",
    style={"width": "50%", "display": "block", "margin": "auto"}),

dcc.Markdown(
    r"""
For example, if we fit the lines with Training Data, then when λ = 0, we could start with the **Least Squares** estimates for the Slope.  
But as we increase the value for **λ**, Ridge and Lasso Regression may shrink **Diet Difference** a lot more than they shrink the Slope.  
"""
),
         
html.H2("Differences"),

dcc.Markdown(
r"""
Alright, we have seen how Ridge and Lasso Regression are similar.  
Now, let's talk about the big difference between them.
"""
),

dcc.Graph(
    id='lasso-lambda-slider-graph',
    figure=lasso_fig,
    style={"width": "120%", "margin": "auto"}
),

dcc.Markdown(
r"""
To see what makes **Lasso** different from **Ridge Regression**, let's go back to the Weight vs Size example, and focus on what happens when we increase the value of **λ**.  

When **λ = 0**, then the **Lasso Regression Line** will be the same as the **Least Squares Line**.  
As **λ** increases in value, the slope gets smaller, until the slope = **0**.  

Therefore, the big difference between Ridge and Lasso is that:  
**Ridge** can only shrink the slope asymptotically close to **0**;
while **Lasso** can shrink the slope all the way to **0**.  

To appreciate this difference, let's look at a big crazy equation:  
- **Size** = y-intercept + slope \* **Weight** + diet difference \* **High Fat Diet** + astrological offset \* **Sign** + airspeed scalar \* **Airspeed of Swallow**  

The term **Weight** and **High Fat Diet** are both reasonable things to use to predict **Size**, but the **Astrological Sign** and **Airspeed of Swallow** are terrible ways to predict **Size**.  

When we apply **Ridge Regression** to this equation, we find the minimal sum of squared residuals plus the **Ridge Regression Penalty**:
- λ * (slope^2 + diet difference^2 + astrological offset^2 + airspeed scalar^2)  

As we increase λ, **slope** and **diet difference** might shrink a little bit; **astrological offset** and **airspeed scalar** might shrink a lot, but they will never be **0**.  

In contrast, with **Lasso Regression**:  
- λ * (|slope| + |diet difference| + |astrological offset| + |airspeed scalar|)  

When we increase λ, **slope** and **diet difference** will shrink a little bit; and **astrological offset** and **airspeed scalar** will go all the way to **0**.  
Then we are left with a way to predict **Size** that only includes **Weight** and **Diet**, and excludes all the silly stuff.   

Since **Lasso Regression** can exclude useless variables from equations, it is a little better than **Ridge Regression** at reducing the **Variance** in models that contain a lot of useless variables.  

In contrast, **Ridge Regression** tends to do a little better when most variables are useful.

---

### Summary

Ridge Regression is very similar to...  
- sum of squared residuals + **λ** \* **the slope^2**  

Lasso Regression  
- sum of squared residuals + **λ** \* **the |slope|**  

As λ increases, both of them **add bias** but **reduce variance**, improving generalization.  

The main difference is that **Lasso Regression** can exclude useless variables from equations, which makes the final equation simpler and eaiser to interpret.

""",
        mathjax=True
    )
])
