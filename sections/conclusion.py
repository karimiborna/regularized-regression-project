from dash import html, dcc

layout = html.Div([
    html.H2("Conclusion and Reflections"),
    dcc.Markdown(
        r"""
Regularization has given us new insights on what makes a regression model “good.”  
In our early work with ordinary least squares (OLS), we focused on getting the **best possible fit** to the data.  
Now we see that a good fit in training might also be a **bad fit for generalization**.  

We need to consider tradeoffs during model selection.  
For example, when choosing between Ridge and Lasso regression, think about what is our goal?  
- Ridge regression shrinks coefficients close to 0, smoothing them out to reduce overfitting. This makes the model more stable, especially when our predictors are correlated or overlapping.  
- Lasso regression takes this a step further by shrinking some coefficients all the way to 0, picking out only the most important features. This helps us focus on what truly matters and makes the model easier to interpret.  

Therefore, Ridge is good for handling multicollinearity and stabilizing the model, while Lasso is better when we need feature selection and a simpler model. Both have their advantages, depending on our goals and data.  

---

### Reflections as Students in Data Science

As students just entered the field linear regression, this feels like an important lesson.  
We are moving beyond the classical approach of fitting lines to data, into the exciting world of machine learning, where the goal isn’t just to understand the past, but to predict the future.  

Regularization is a first step into this new way of thinking:  
- It’s not just about fitting the data — it’s about **generalizing** from what we've seen to what we might face next.  
- Sometimes it is okay to accept some bias because it might help reduce variance and unpredictability of our models.  
- We should carefully select models that will work with messy real-world data, not just perfect training sets.  

---

### Why This Matters for Our Field

In the real world—whether it’s biomedical research, finance, or social science—data can get really complex. There are often thousands of variables that might be noisy and overlap with each other...  

Just using ordinary least squares (OLS) regression won’t be enough here — it tends to fall apart when the data gets more and more complicated. That’s where Ridge and Lasso come in. They provide a better way to manage this complexity and keep our models from overfitting:  
- In genomics, there might be thousands of genes that could be used for predicting a disease. Then we can use Lasso to help focus on a selection of genes that trully matter.  
- In finance, markets are often unstable and many features are correlated. Ridge regression can help stabilize predictions so they’re more reliable.  
- In almost any kind of scientific research, regularization helps us avoid looking for noise in the data and instead find important patterns that matter.  

---

### Final Thoughts

What really excites us is that this is just the **beginning** of our journey.  
Regularization sits at the edge between traditional regression and the powerful machine learning techniques we're about to dive into.  

We've learned that adding more parameters and chasing a perfect fit doesn’t always mean better science.
We've also learned that adding thoughtful penalties can make our models smarter and simpler.  

From this project, we not only learned a new method, but also a whole new way of thinking about modeling:  
It’s not just about fitting the past perfectly, but about predicting the future right.
""",
        mathjax=True
    )
])
