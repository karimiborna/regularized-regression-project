# Regularized Regression Project
Final project for Fall 2025 Linear Regression Analysis.  
The live blog is [here](https://regularized-regression-project.onrender.com/).  

# Regularized Regression Dashboard

This project is an interactive dashboard built with [Plotly Dash](https://dash.plotly.com/) to demonstrate the concepts of **Ridge** and **Lasso** regression.  

It was developed as part of a master's-level course in data science and serves as both an educational tool and a project deliverable.

---

## Features
- **Introduction & Lessons**:  
  Gives a walkthrough of what linear, Ridge, and Lasso regression actually do.  
  Uses math (with LaTeX rendering) and visuals to explain how overfitting happens and how regularization helps fix it.

- **Data Preview**:  
  Shows the dataset used in the project (the Diabetes dataset from scikit-learn) so you can see what features we are working with before fitting any models.

- **Performance & Model Fit**:  
  Lets you interact with sliders to see how changing λ (lambda) affects Ridge and Lasso regression.  
  You can watch the model predictions and errors update in real time to get a feel for how regularization changes the fit.

- **Coefficient Paths**:  
  Plots how each model coefficient moves toward zero as λ increases.  
  You can really see the difference here: Ridge shrinks coefficients smoothly, while Lasso drives some all the way to zero.

- **Bias–Variance Tradeoff**:  
  Interactive plot that shows how bias, variance, and MSE change with λ.  
  It is a simple way to see why there is a "sweet spot" for regularization. Too little and you overfit, too much and you underfit.

- **Conclusion & Reflections**:  
  Wraps up what we learned about regularization and why it matters.  
  Talks about how Ridge and Lasso help models generalize better and what tradeoffs come with tuning λ.

- **About & References**:  
  Explains the purpose of the project, who built it, and lists the main resources we used.  
  Includes course materials, academic papers, and a few online articles that helped with intuition and visuals.

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


