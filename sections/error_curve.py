from dash import html, dcc
import dash
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

layout = html.Div([
    html.H2("Train vs Test Error"),
    html.P("This plot shows how training error and test error vary with λ, illustrating the bias–variance tradeoff."),
    dcc.Dropdown(
        id="error-model-type",
        options=[{"label": "Ridge", "value": "ridge"}, {"label": "Lasso", "value": "lasso"}],
        value="ridge"
    ),
    dcc.Graph(id="error-curve-plot")
])

def register_callbacks(app):
    @app.callback(
        dash.Output("error-curve-plot", "figure"),
        dash.Input("error-model-type", "value")
    )
    def update_error_plot(model_type):
        lambdas = np.logspace(-2, 2, 30)
        train_err, test_err = [], []

        for alpha in lambdas:
            if model_type == "ridge":
                model = Ridge(alpha=alpha)
            else:
                model = Lasso(alpha=alpha, max_iter=5000)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_err.append(mean_squared_error(y_train, y_pred_train))
            test_err.append(mean_squared_error(y_test, y_pred_test))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.log10(lambdas), y=train_err, mode="lines", name="Train Error"))
        fig.add_trace(go.Scatter(x=np.log10(lambdas), y=test_err, mode="lines", name="Test Error"))

        fig.update_layout(
            title=f"Train vs Test Error ({model_type.title()})",
            xaxis_title="log10(λ)",
            yaxis_title="Mean Squared Error"
        )
        return fig
