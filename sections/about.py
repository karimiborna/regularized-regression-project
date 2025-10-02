from dash import html

layout = html.Div([
    html.H2("About This Project"),
    html.P("This dashboard was created for the Final Project in the Linear Regression Analysis class at USFCA. "
           "It demonstrates how Ridge and Lasso regression apply penalties to coefficients "
           "to reduce overfitting and improve generalization."),
    html.P("Author: Borna Karimi, Alan Luo"),
    html.P("Built with: Python, scikit-learn, Plotly Dash")
])
