from dash import html, dcc
import dash
import numpy as np
import plotly.graph_objects as go

# ------------------------------------------------------------
# Generate bias, variance, and MSE functions (simulated)
# ------------------------------------------------------------
def bias_variance_decomposition(lam):
    # Simulated smooth curves that depend on lambda
    bias_sq = 0.05 + 0.02 * lam
    variance = 0.8 * np.exp(-0.25 * lam)
    noise = 0.1                           # irreducible noise
    mse = bias_sq + variance + noise
    return bias_sq, variance, noise, mse

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
layout = html.Div([
    html.H2("Bias–Variance Tradeoff"),
    html.P(
        "This interactive plot shows how bias, variance, and mean squared error (MSE) change "
        "as the regularization parameter λ increases. "
        "As λ grows, the model becomes simpler — bias increases but variance decreases. "
        "The goal is to find the λ that minimizes MSE, which balances the two."
    ),

    html.P(
        "Before we get into our interactive graph, lets recap on some key concepts:"
    ),

    dcc.Markdown(
        r"""

Remember the formula for MSE, which we often use to measure the performance of unseen data.

$$
\text{MSE} = \mathbb{E}\left[(y - \hat{y})^2\right]
$$

That MSE can be broken into components where:

Bias² = on average, how far your model's predictions are from the true function. 
This measures oversimplification (keep in mind, THIS MEASURES WHEN ITS TOO SIMPLE)

Variance = how much your model's predictions change across training samples. 
This measures overfitting (keep in mind, THIS MEASURES WHEN ITS TOO FLEXIBLE)
$$
\text{MSE}(x) = \text{Bias}[\hat{y}]^2 + \text{Var}[\hat{y}]
$$
So as lambda increases: 
-The model gets simpler (coefficients shrink).
-Variance decreases.
-Bias increases and the model is less capable of capturing the true signal.

Lets connect this back to ridge specifically:

$$
\hat{\beta}_{\text{ridge}} = (\mathbf{X}'\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}'\mathbf{Y}
$$

with a λ = 0, this is equal to OLS, which has high variance when predictors are correlated.

When λ approaches infinity, Beta approaches 0, which produces high bias. This, in turn, causes our predictions to collapse to the mean.

Between these extremes, Ridge finds a Beta that minimizes total MSE. Ideally, it reduces variance more than increasing bias. Why?

Remember that high bias makes a model incapable of capturing true signal.

    """,
        mathjax=True
    ),

    html.Label("Adjust λ (Regularization Strength):"),

    dcc.Slider(
        id="lambda-slider",
        min=0,
        max=25,
        step=0.1,
        value=1,
        marks={i: str(i) for i in range(0, 11, 2)},
        tooltip={"always_visible": False}
    ),

    dcc.Graph(id="bias-variance-plot"),

    html.P(
        "The blue curve shows the variance term — how sensitive the model is to the training data. "
        "The red curve shows bias squared — how far the average prediction is from the true function. "
        "Their sum, the green curve (MSE), has a minimum at the ideal λ."
    ),

    html.P(
        "As you will see in the graph:" \
        "With a small λ (≈0), the bias is low, the variance is high, and the variance overfits. "
        "With a moderate λ, the  bias is slightly higher, the variance is lower, and the MSE is minimized (in this graph, around 9)"
        "With a large λ, the bias is very high, the variance is very low, which means the MSE underfits."
    
    )
])

# ------------------------------------------------------------
# Callback
# ------------------------------------------------------------
def register_callbacks(app):
    @app.callback(
        dash.Output("bias-variance-plot", "figure"),
        dash.Input("lambda-slider", "value")
    )
    def update_plot(lam):
        lam_values = np.linspace(0, 25, 150)
        bias_vals, var_vals, mse_vals, noise_vals = [], [], [], []

        for l in lam_values:
            b, v, n, m = bias_variance_decomposition(l)
            bias_vals.append(b)
            var_vals.append(v)
            noise_vals.append(n)
            mse_vals.append(m)

        # Compute current values for selected λ
        current_b, current_v, current_n, current_m = bias_variance_decomposition(lam)

        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lam_values, y=bias_vals, mode="lines", name="Bias²",
            line=dict(color="red", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=lam_values, y=var_vals, mode="lines", name="Variance",
            line=dict(color="blue", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=lam_values, y=mse_vals, mode="lines", name="MSE",
            line=dict(color="green", width=3)
        ))

        # Vertical line for selected λ
        fig.add_vline(x=lam, line=dict(color="gray", dash="dot"))

        # Text annotation with numerical values
        annotation_text = (
            f"λ = {lam:.2f}<br>"
            f"Bias² = {current_b:.3f}<br>"
            f"Variance = {current_v:.3f}<br>"
            f"MSE = {current_m:.3f}"
        )

        fig.add_annotation(
            x=lam,
            y=current_m,
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            yshift=10,
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1
        )

        fig.update_layout(
            title="Bias–Variance–MSE as a Function of λ",
            xaxis_title="λ (Regularization Strength)",
            yaxis_title="Error Component Magnitude",
            legend=dict(x=0.7, y=0.95),
            height=500
        )

        return fig

