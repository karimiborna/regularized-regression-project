from dash import html, dcc
import dash
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ------------------------------------------------------
# Generate simple multiple linear regression data
# ------------------------------------------------------
X, y = make_regression(
    n_samples=120, n_features=6, noise=15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------------------------------------
# Layout
# ------------------------------------------------------
layout = html.Div([
    html.H2("Predicted vs Actual Values"),
    html.P(
        "This plot shows how predictions become less responsive as the "
        "regularization strength λ increases. Small λ values fit the data "
        "closely (low bias, high variance), while large λ values flatten "
        "the relationship (high bias, low variance)."
    ),

    html.Label("Select Model:"),
    dcc.Dropdown(
        id="pred-model-type",
        options=[
            {"label": "Ridge Regression", "value": "ridge"},
            {"label": "Lasso Regression", "value": "lasso"}
        ],
        value="ridge",
        style={"width": "300px"}
    ),

    html.Label("Lambda (α):"),
    dcc.Slider(
        id="pred-alpha-slider",
        min=0.01, max=50, step=0.5, value=1,
        marks={i: str(i) for i in [0, 1, 5, 10, 20, 50]},
        tooltip={"always_visible": False, "placement": "bottom"}
    ),

    dcc.Graph(id="pred-vs-actual-plot"),
    html.Div(id="metrics-output", style={"textAlign": "center", "marginTop": 20})
])

# ------------------------------------------------------
# Callbacks
# ------------------------------------------------------
def register_callbacks(app):
    @app.callback(
        [dash.Output("pred-vs-actual-plot", "figure"),
         dash.Output("metrics-output", "children")],
        [dash.Input("pred-model-type", "value"),
         dash.Input("pred-alpha-slider", "value")]
    )
    def update_plot(model_type, alpha):
        # Train model
        model = Ridge(alpha=alpha) if model_type == "ridge" else Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Build scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode="markers",
            name="Predicted vs Actual",
            marker=dict(size=6, color="#1f77b4")
        ))
        fig.add_trace(go.Scatter(
            x=y_test, y=y_test,
            mode="lines",
            name="Ideal: y = ŷ",
            line=dict(color="black", dash="dot")
        ))

        fig.update_layout(
            title=f"{model_type.title()} Regression (λ = {alpha:.2f})",
            xaxis_title="True y",
            yaxis_title="Predicted y",
            height=500,
            margin=dict(l=60, r=40, t=60, b=50)
        )

        metrics_text = f"Test R² = {r2:.3f}  MSE = {mse:.2f}"
        return fig, metrics_text
