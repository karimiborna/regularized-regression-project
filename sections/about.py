from dash import html

layout = html.Div([
    html.H2("About This Project"),
    html.P("This dashboard was for the Final Project of F25 Linear Regression Analysis class at USFCA."),
        html.P("It demonstrates how Ridge and Lasso regression apply penalties to coefficients "
           "to reduce overfitting and improve generalization."),
    html.P("Author: Borna Karimi, Alan Luo"),
    html.P("Built with: Python, scikit-learn, Plotly Dash")
])
