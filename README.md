# regularized-regression-project
final project for Linear Regression Analysis.
# Regularized Regression Dashboard

This project is an interactive dashboard built with [Plotly Dash](https://dash.plotly.com/) to demonstrate the concepts of **Ridge** and **Lasso** regression.  
It was developed as part of a master's-level course in data science and serves as both an educational tool and a project deliverable.

---

## Features
- **Introduction & Lessons**: Narrative explanations with math (LaTeX rendered inline).
- **Data Preview**: Display of the dataset (Diabetes dataset from scikit-learn).
- **Model Fit**: Interactive Ridge/Lasso regression with λ slider and coefficient plots.
- **Coefficient Paths**: How model coefficients evolve as λ increases.
- **Train vs Test Error Curve**: Visualization of the bias–variance tradeoff.
- **Residual Plot**: Model diagnostics to evaluate generalization.
- **Conclusion & Reflections**: Student perspective on why regularized regression matters.

---

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/regularized-regression-dashboard.git
   cd regularized-regression-dashboard

2. Create and activate a virtual environment:
  ```bash
  conda create -n regdash python=3.10
  conda activate regdash
  ```
  or with venv:
  ```bash
  python -m venv venv
  source venv/bin/activate   # macOS/Linux
  venv\Scripts\activate      # Windows
  ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open your browser at:
   ```bash
   http://127.0.0.1:8050/
   ```
## Technologies
- Python 3.10
- Dash (Plotly)
- Plotly
- scikit-learn
- Pandas
- NumPy
- Gunicorn (for deployment)

## Authors
Borna Karimi, Alan Luo
MS Data Science Candidates
University of San Francisco


