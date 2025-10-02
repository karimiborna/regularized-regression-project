import dash
from dash import html
from sections import (
    introduction, lessons, performance, conclusion,
    data_preview, model_fit, about,
    coeff_path, error_curve, residuals
)

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Custom page-wide CSS for a centered, compact look
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      body {
        font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
        font-size: 15px;  /* smaller base font */
        line-height: 1.5;
      }
      h1 {
        text-align: center;
        font-size: 28px;
        margin-bottom: 20px;
      }
      h2 {
        text-align: center;
        font-size: 22px;
        margin-top: 35px;
        margin-bottom: 15px;
      }
      p {
        text-align: center;
      }
      .section {
        margin-bottom: 40px;  /* space between sections */
      }
      .dash-graph {
        margin: 0 auto;
        max-width: 700px;  /* shrink plots */
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

# Single-page layout (sections stacked, no separators)
app.layout = html.Div([
    html.H1("Regularized Regression Dashboard"),

    html.Div(introduction.layout, className="section"),
    html.Div(lessons.layout, className="section"),
    html.Div(data_preview.layout, className="section"),
    html.Div(model_fit.layout, className="section"),
    html.Div(coeff_path.layout, className="section"),
    html.Div(error_curve.layout, className="section"),
    html.Div(residuals.layout, className="section"),
    html.Div(performance.layout, className="section"),
    html.Div(conclusion.layout, className="section"),
    html.Div(about.layout, className="section")
])

# Register callbacks for interactive sections
model_fit.register_callbacks(app)
coeff_path.register_callbacks(app)
error_curve.register_callbacks(app)
residuals.register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
