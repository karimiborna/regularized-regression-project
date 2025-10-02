from dash import html, dcc
import dash
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso

X, y = load_diabetes(return_X_y=True, as_frame=True)
feature_names = load_diabetes().feature_names

layout = html.Div([
    html.H2("Coefficient Paths"),
    html.P("This plot shows how coefficients change as λ increases. "
           "Ridge shrinks coefficients smoothly, while Lasso drives some to zero abruptly."),
    dcc.Dropdown(
        id="path-model-type",
        options=[{"label": "Ridge", "value": "ridge"}, {"label": "Lasso", "value": "lasso"}],
        value="ridge"
    ),
    dcc.Graph(id="coeff-path-plot")
])

def register_callbacks(app):
    @app.callback(
        dash.Output("coeff-path-plot", "figure"),
        dash.Input("path-model-type", "value")
    )
    def update_path_plot(model_type):
        lambdas = np.logspace(-2, 2, 30)
        coeffs = []

        for alpha in lambdas:
            if model_type == "ridge":
                model = Ridge(alpha=alpha)
            else:
                model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(X, y)
            coeffs.append(model.coef_)

        coeffs = np.array(coeffs).T

        fig = go.Figure()
        for i, feat in enumerate(feature_names):
            fig.add_trace(go.Scatter(x=np.log10(lambdas), y=coeffs[i], mode="lines", name=feat))

        fig.update_layout(
            title=f"Coefficient Paths ({model_type.title()})",
            xaxis_title="log10(λ)",
            yaxis_title="Coefficient Value"
        )
        return fig
