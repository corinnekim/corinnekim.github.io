---
layout: single
title: "Inferential Statistics - Standard Error, Confidence Interval, T-score"
date: 2025-12-09
categories: [statistics]
tags: [standard error, confidence interval, t-score]
math: true
---
## Inferential Statistics

* `What we want to know`: &nbsp;the population mean and population standard deviation
* `What we already know`: &nbsp;the sample size, sample mean, and sample standard deviation   
    * Theoretically, we assume that we draw sample means many times.
    * The distribution of these sample means follows a normal distribution with mean $\mu$ and standard deviation $\frac{\sigma}{\sqrt{n}}$ (= **Standard Error**). 

    $$
    \bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)
    $$

    * `Central Limit Theorem (CLT):` Even if the population isn’t normally distributed, the sampling distribution of the mean becomes **approximately normal** once our sample size $n$ is large enough (usually $n \ge 30$).    
<br>

* **Sample error** ($\bar{X}$ - $\mu$) follows a normal distribution with a mean 0 and a standard deviation $\frac{\sigma}{\sqrt{n}}$ as the sample size grows.

    $$
    (\bar{X} - \mu) \sim N\left(0, \frac{\sigma^2}{n}\right)
    $$

<br>
## Z-score, t-distribution
* **Z-score:** indicates how many **standard errors** the sample mean is away from the population mean.

    $$
    Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}
    $$


* In practice, however, we usually don't know the population's standard deviation $\sigma$.   
    * So, we replace it with the sample standard deviation $s$.
* In that case, we use the **t–score** instead of the **Z–score**.  

    $$
    t = \frac{\bar{X} - \mu}{s / \sqrt{n}}
    $$

    * **`t-distribution`**: Since $s$ varies across samples, the calculated t-scores carry more uncertainty than Z-scores.
    * The distribution of the t–statistic therefore spreads out more than the normal distribution. It has a similar shape but heavier tails.   

`Interpretation`: the t–value represents how many **standard errors** the sample mean is from the hypothesized population mean.
<br>
<br>

## Confidence Interval
<br>
![](/assets/images/statistically_confident.jpeg){: width= "60%"}
<br>
* **Confidence Interval Formula:**

$$
\left( \bar{x} - \text{threshold} \times \text{SE}, \quad \bar{x} + \text{threshold} \times \text{SE} \right)
$$

When we know $\sigma$: &nbsp; $\text{threshold} = \mathbf{z\text{-score}}$ (e.g., 1.96)

When we don't know $\sigma$: &nbsp; $\text{threshold} = \mathbf{t\text{-score}}$ (e.g., 2.045)

*(The **threshold** is defined by our confidence level.)*


* **Example:** `n = 30`, &nbsp; `df = 29`, &nbsp; `95% Confidence Interval` (assuming t-distribution)   
*(df = degrees of freedom, here it’s `n−1`)*

$$
 \left( \bar{x} - 2.045 \times \frac{s}{\sqrt{n}}, \quad \bar{x} + 2.045 \times \frac{s}{\sqrt{n}} \right)
$$


`What this means`: With many repeated samples, about 95% of the confidence intervals you’d build would contain the true population mean.

<!-- There is a 95% probability that the interval covers **the true population mean** ($\mu$).     -->

<!-- ![](/assets/images/regina_george.jpeg){: width="60%"}
<br> -->

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']]
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>