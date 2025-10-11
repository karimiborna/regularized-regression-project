from dash import html

layout = html.Div([
    html.H2("About This Project"),
    
    html.P("This dashboard was for the Final Project of F25 Linear Regression Analysis class at USFCA."),
    
    html.P("It demonstrates how Ridge and Lasso regression apply penalties to model coefficients "
           "to reduce overfitting, improve generalization, and illustrate key concepts such as "
           "the bias–variance tradeoff and regularization geometry."),
    
    html.P("Author: Borna Karimi, Alan Luo"),
    
    html.P("Built with: Python, matplotlib, scikit-learn, and Plotly Dash"),

    html.H3("References"),

    html.Ul([
        html.Li([
            html.Span("Hoerl, A. E., & Kennard, R. W. (1970). "),
            html.I("Ridge regression: Biased estimation for nonorthogonal problems. "),
            "Technometrics, 12(1), 55–67. ",
            html.A("https://doi.org/10.1080/00401706.1970.10488634",
                   href="https://doi.org/10.1080/00401706.1970.10488634", target="_blank")
        ]),
        html.Li([
            html.Span("Tibshirani, R. (1996). "),
            html.I("Regression shrinkage and selection via the lasso. "),
            "Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267–288. ",
            html.A("https://doi.org/10.1111/j.2517-6161.1996.tb02080.x",
                   href="https://doi.org/10.1111/j.2517-6161.1996.tb02080.x", target="_blank")
        ]),
        html.Li([
            html.Span("Penn State University. (n.d.). "),
            html.I("Applied Data Mining and Statistical Learning (STAT 897D): Ridge Regression. "),
            html.A("https://online.stat.psu.edu/stat857/node/137/",
                   href="https://online.stat.psu.edu/stat857/node/137/", target="_blank")
        ]),
        html.Li([
            html.Span("Parr, T. (n.d.). "),
            html.I("Regularization explained simply. "),
            html.A("https://explained.ai/regularization/",
                   href="https://explained.ai/regularization/", target="_blank")
        ]),
        html.Li([
            html.Span("Stack Exchange (Cross Validated). (n.d.). "),
            html.I("Derivation of the closed-form Lasso solution. "),
            html.A("https://stats.stackexchange.com/questions/17781/derivation-of-closed-form-lasso-solution",
                   href="https://stats.stackexchange.com/questions/17781/derivation-of-closed-form-lasso-solution",
                   target="_blank")
        ]),
        html.Li([
            html.Span("Tibshirani, R. (2023). "),
            html.I("High-Dimensional Regression: Ridge. "),
            "Advanced Topics in Statistical Learning, Spring 2023. Carnegie Mellon University. "
            "Retrieved from uploaded course material (ridge.pdf)."
        ])
    ], style={"lineHeight": "1.8"})
])
