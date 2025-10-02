from dash import html, dcc
import dash
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

layout = html.Div([
    html.H2("Residual Plot"),
    html.P("Residuals should look random if the model generalizes well. "
           "This plot shows residuals vs fitted values as λ changes."),
    dcc.Dropdown(
        id="resid-model-type",
        options=[{"label": "Ridge", "value": "ridge"}, {"label": "Lasso", "value": "lasso"}],
        value="ridge"
    ),
    dcc.Slider(
        id="resid-alpha-slider",
        min=-2, max=2, step=0.1, value=0,
        marks={i: f"1e{i}" for i in range(-2, 3)}
    ),
    dcc.Graph(id="resid-plot")
])

def register_callbacks(app):
    @app.callback(
        dash.Output("resid-plot", "figure"),
        [dash.Input("resid-model-type", "value"),
         dash.Input("resid-alpha-slider", "value")]
    )
    def update_resid_plot(model_type, log_alpha):
        alpha = 10**log_alpha

        if model_type == "ridge":
            model = Ridge(alpha=alpha)
        else:
            model = Lasso(alpha=alpha, max_iter=5000)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", name="Residuals"))
        fig.update_layout(
            title=f"Residuals vs Fitted Values ({model_type.title()}, λ={alpha:.3f})",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            shapes=[dict(type="line", x0=min(y_pred), x1=max(y_pred), y0=0, y1=0, line=dict(dash="dot"))]
        )
        return fig
