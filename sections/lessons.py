from dash import html, dcc

layout = html.Div([
    html.H2("Lessons Learned: Ridge vs. Lasso 🤓"),
    dcc.Markdown(
        r"""
### Ridge Regression (L2 penalty)

Ridge minimizes the **penalized SSE**:

$$
\mathcal{L}({\beta})
= \sum_{i=1}^{n} \big(y_i - \hat{y}_i\big)^2
+ \lambda \sum_{j=1}^{p} \beta_j^2
$$

- The penalty \( \lambda \sum_{j=1}^{p} \beta_j^2 \) shrinks coefficients toward zero.  
- Coefficients never become exactly zero.  
- Works well when predictors are correlated and all carry some signal.  

---

### Lasso Regression (L1 penalty)

Lasso minimizes:

$$
\mathcal{L}({\beta})
= \sum_{i=1}^{n} \big(y_i - \hat{y}_i\big)^2
+ \lambda \sum_{j=1}^{p} \lvert \beta_j \rvert
$$

- The L1 penalty encourages **sparsity**.  
- Some coefficients shrink all the way to zero (feature selection).  
- Useful when only a subset of predictors truly matter.  

---

### Key Lesson

- **Ridge**: smooth shrinkage — all coefficients decrease but remain in the model.  
- **Lasso**: selection — some coefficients drop to exactly zero.  

As \( \lambda \) increases, both add bias but reduce variance, improving generalization.  
""",
        mathjax=True
    )
])
