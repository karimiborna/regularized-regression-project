from dash import html, dash_table
import pandas as pd
from sklearn.datasets import load_diabetes

# Load example dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)
df = X.copy()
df["target"] = y

layout = html.Div([
    html.H2("Data Preview"),
    html.P("We use the Diabetes dataset included in scikit-learn. It contains 10 baseline features "
           "such as age, BMI, and blood pressure, along with a target variable measuring disease progression."),
    dash_table.DataTable(
        data=df.head(15).to_dict("records"),
        columns=[{"name": col, "id": col} for col in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "5px"},
        page_size=10
    )
])
