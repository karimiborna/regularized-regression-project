from dash import html

layout = html.Div([
    html.H2("Model Performance and Limitations"),
    html.P("Regularization introduces a tradeoff between bias and variance. "
           "As λ increases, training error usually increases because the model is more restricted, "
           "but test error often decreases because the model generalizes better."),
    html.H3("Choosing Between Ridge and Lasso"),
    html.Ul([
        html.Li("Use Ridge when predictors are many and correlated."),
        html.Li("Use Lasso when you want a simpler model and automatic feature selection.")
    ]),
    html.H3("Limitations"),
    html.Ul([
        html.Li("Choosing the right λ is critical. Too small, and overfitting persists; too large, and the model underfits."),
        html.Li("Ridge keeps all predictors, which can be harder to interpret in high dimensions."),
        html.Li("Lasso can behave inconsistently when variables are strongly correlated.")
    ])
])
