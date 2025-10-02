from dash import html, dcc
import plotly.graph_objects as go
import numpy as np
import dash
from sklearn.linear_model import Ridge, Lasso

# Synthetic data generator
rng = np.random.RandomState(42)
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y_true = np.sin(X).ravel()
y = y_true + rng.normal(scale=0.3, size=X.shape[0])  # add noise

layout = html.Div([
    html.H2("Model Fit & Overfitting Visualization"),
    html.P("This demo shows how regularization reduces overfitting. "
           "As λ increases, the fitted line becomes smoother and less sensitive to noise."),
    
    html.Label("Select Model:"),
    dcc.Dropdown(
        id="model-dropdown",
        options=[
            {"label": "Ridge Regression", "value": "ridge"},
            {"label": "Lasso Regression", "value": "lasso"}
        ],
        value="ridge"
    ),

    html.Label("Lambda (α):"),
    dcc.Slider(
        id="alpha-slider",
        min=0.01, max=50, step=0.5, value=1,
        marks={i: str(i) for i in [0, 1, 5, 10, 20, 50]}
    ),

    dcc.Graph(id="fit-plot")
])

def register_callbacks(app):
    @app.callback(
        dash.Output("fit-plot", "figure"),
        [dash.Input("model-dropdown", "value"),
         dash.Input("alpha-slider", "value")]
    )
    def update_plot(model_type, alpha):
        # Polynomial features to force potential overfitting
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        poly = PolynomialFeatures(degree=12)  # high degree → can overfit
        if model_type == "ridge":
            model = make_pipeline(poly, Ridge(alpha=alpha))
        else:
            model = make_pipeline(poly, Lasso(alpha=alpha, max_iter=5000))

        model.fit(X, y)
        y_pred = model.predict(X)

        # Plot data + fitted curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.ravel(), y=y, mode="markers", name="Noisy Data"))
        fig.add_trace(go.Scatter(x=X.ravel(), y=y_pred, mode="lines", name=f"Fitted {model_type.title()}"))
        fig.add_trace(go.Scatter(x=X.ravel(), y=y_true, mode="lines", name="True Function", line=dict(dash="dot")))
        
        fig.update_layout(title="Effect of Regularization on Overfitting",
                          xaxis_title="X",
                          yaxis_title="y")
        return fig
