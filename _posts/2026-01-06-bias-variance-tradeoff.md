---
layout: single
title: "Machine Learning Fundamentals: Bias and Variance Tradeoff"
date: 2026-01-06
categories: [datascience]
tags: [machine learning]
math: true
toc: true
toc_sticky: true
---
## Bias-Variance Tradeoff

In this post, I’ll break down a core ML concept: the bias–variance tradeoff.
<br>

Here’s a toy example with three complexity levels. I sampled 30 random points from a true function (orange line) and fit polynomial regressions at three degrees to learn the relationship between x and y.
<br>

![](/assets/images/bias-variance/output.png){: width='100%'}

**Degree 1** (a simple linear regression) fits a straight line. Since the true relationship is curved, the line consistently misses it. That kind of systematic error is **bias** — the model is too simple to represent the pattern.
<br>

**Degree 15** fits a very wiggly curve. It’s flexible enough to hug the training points closely, so it has **low bias**. However, that flexibility could turn into <span style="background-color: #fff5b1">"memorizing"</span> the train data. If a few points change or get left out during cross-validation, the curve can swing wildly and the error can blow up.
<br>

**Degree 4** sits in the middle. It’s flexible enough to capture most of the curve, but not so flexible that it starts chasing every noisy point. In this example, it ends up closest to the <span style="background-color: #fff5b1">sweet spot</span>: low enough bias to learn the pattern, without the instability that comes with high variance.

The MSE reflects that. Degree 1 lands around 0.4, degree 4 drops to ~0.04, and degree 15 blows up to `1.83e8`. That’s **overfitting**: it can look great on the training set, but it becomes unstable and unreliable on new data.
<br>

In contrast, the straight line is more stable across datasets (**lower variance**), but it consistently misses the real curve (**higher bias**), which is **underfitting**.
<br>

| Degree | Bias | Variance | Fits training | MSE | Outcome |
|:------:|:----:|:--------:|:------------:|:---:|:-------:|
|   1    | High |   Low    |     Poor     | 0.4 | Underfitting |
|   4    | Low  | Medium   |     Good     | 0.04 | Best generalization |
|   15   | Very low  |   High   |  Very good   | `1.83e8` | Overfitting |



## Complexity and Error

![](/assets/images/bias-variance/bias-variance-total-error.png){: width='80%'}

<br>
As a model gets more complex, bias and variance usually move in opposite directions: bias(<span style="color:#e74c3c;">the red curve</span>) drops, and variance(<span style="color:#0f766e;">the teal curve</span>) rises. That’s why the total error (the black curve) often looks U-shaped. 
<br>

As you increase complexity, error often drops at first. But if you keep going, variance starts to dominate and total error rises again.

- On the left side, the model is too simple, so it underfits — <span style="background-color: #fff5b1">high bias, low variance.</span>  
- On the right side, the model is too flexible, so it overfits — <span style="background-color: #fff5b1">low bias, high variance.</span>

What we want is the dip in the middle — the “Goldilocks” zone — where bias and variance balance out and total error is lowest. The goal in ML is to find that **sweet spot**, where the model learns the general pattern without overfitting the noise.

## Finding the Sweet Spot
So how do we find the sweet spot?      
<br>
In practice, that comes down to three things: (1) estimate validation error, (2) keep variance under control, and (3) use learning curves to tell whether you’re dealing with bias or variance.    
<br>
**Cross-validation (CV)**     
CV splits the training data into 𝑘 folds (often 5 or 10) and evaluates the model 𝑘 times, each time holding out a different fold.
It **averages the 𝑘 scores**, which helps you pick hyperparameters that perform well across folds and generalize better to new data.
 
**Regularization**    
High variance often happens when the model uses really large coefficients to fit the training points too closely.     
L1/L2 regularization adds a **penalty** for large coefficients, so the model keeps them smaller.

**Learning curve**   
Plot training and validation error as you increase the number of training examples.
<br>

![](/assets/images/bias-variance/learning-curve-underfit.png){: width='70%'}

If both curves <span style="background-color: #fff5b1">stay high and close,</span> the model has high bias (underfitting).      
It can’t fit the training set well and adding more data doesn’t help much.

![](/assets/images/bias-variance/learning-curve-overfit.png){: width='70%'}

If <span style="background-color: #fff5b1">training error is low but validation error stays much higher,</span> the model is fitting noise (overfitting).     
It’s doing well on the training set, but that performance doesn’t carry over to new data.

![](/assets/images/bias-variance/learning-curve-ideal.png){: width='70%'}

Ideally, the model fits the training data reasonably well and also generalizes to validation data. As you add more samples, <span style="background-color: #fff5b1">both curves typically stabilize and stay close.</span>


## Summary
- Generalization error is shaped by bias, variance, and noise.
- Bias is the model’s average error across different training sets.
- Variance is how sensitive the model is to changes in the training set.
- There’s no perfect model — <span style="background-color: #fff5b1">the job is finding the one with the right trade-off for your data.</span>