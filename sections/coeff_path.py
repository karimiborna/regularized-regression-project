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
           "Ridge shrinks coefficients smoothly, while Lasso drives some to zero abruptly. "
           "Consider for Ridge: The penalty term is squared, which makes the coefficients drop smoothly as lambda increases. "
           "The shrinkage is uniform and continuous: big coefficients shrink faster, small ones slower, but they all stay in the model. "
    ),

    html.P(
           "However, for Lasso: The penalty term is based on absolute magnitude. "
           "Absolute value can create sharp corners, up to the point, in some data, of shaping like a diamond. "
           "As lambda increases, some coefficients hit exactly zero and stay there, the model effectively drops those predictors. "
           "This is how lambda does feature selection. As predictors drop, fewer are left."
           "The result is a sparser model!"
    ),
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
