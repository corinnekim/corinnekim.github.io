---
layout: single
title: "Understanding Time Series Forecasting with ARIMA"
date: 2026-01-14
categories: [datascience]
tags: [machine learning, timeseries-data]
toc: true
toc_sticky: true
---

<!-- <span style="background-color: #fff5b1">text</span> -->
![](/assets/images/arima/arima.png){: width='80%'}

**ARIMA** is a fundamental model in time-series analysis. Let’s break it down and practice the workflow with Python.

## What is ARIMA?
ARIMA stands for AutoRegressive Integrated Moving Average. The basic idea is to <span style="background-color: #fff5b1">separate what’s predictable from what’s basically random.</span>     
<br>
Any time series $y_t$ has two parts:
1. a part you can model (trend, seasonality, cycles, etc.)    
2. and a part you can’t (random shocks)     

## The ARIMA workflow
A typical ARIMA workflow looks like this:           
1. Run EDA and check if the series is **stationary**.
2. If it’s not, difference the series to remove trend/seasonality (the **‘I’** in ARIMA)    
3. Fit AR(𝑝) and MA(𝑞) terms on the stationary series.    

After fitting ARIMA, the residuals should look **random**. If the errors still show a pattern, that means some structure is still left. Ideally, what remains should be close to white noise.

## AR MA models and I
AR and MA look at the current value from two different angles. 
### 1. AR (Autoregressive) model: past values shape future  

AR works when today tends to follow yesterday and the past explains the present. That’s the **autoregressive** part of a time series. You see this in <span style="background-color: #fff5b1">temperature, prices, demand, or something that accumulates over time.</span>

$$
y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t
$$

The present $y_t$ is a weighted mix of its own past (from $y_{t-1}$ to $y_{t-p}$), plus a random noise $\varepsilon_t$.    
$p$ tells you how many past values the model looks back on.

### 2. MA (Moving Average) model: past shocks linger
Unlike AR, which looks at past values, MA looks at past shocks. We use it when sudden events matter more than the level itself, like <span style="background-color: #fff5b1">supply shortages, factory outages, or sudden policy announcements.</span>    

$$
y_t = c + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}
$$

The present $y_t$ is a weighted mix of past shocks $\varepsilon_{t-1} + \varepsilon_{t-2} + \cdots + \varepsilon_{t-q}$  plus a new shock $\varepsilon_t$.    
$q$ means how many past shocks it keeps in the mix.


> A common misunderstanding: The **'moving average'** in MA is not a rolling mean. It’s a weighted mix of past errors.

For example, in an MA(2) model,

$$
y_t = \mu + \varepsilon_t + 0.6\,\varepsilon_{t-1} - 0.2\,\varepsilon_{t-2}
$$

Most of the influence comes from yesterday’s shock $\varepsilon_{t-1}$ (0.6), with a smaller negative effect from two days back $\varepsilon_{t-2}$ (−0.2).

### ARMA(AR + MA) Model
ARMA is when you use both AR and MA.
For example, <span style="background-color: #fff5b1">inflation or unemployment often show both continuing value patterns(AR) and lingering shocks after a policy change or a market shock(MA).</span> A common sign for ARMA is that both ACF and PACF tail off rather than cut off.

$$
y_t = \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}
$$


### I (Integrated)
ARMA assumes the series is stationary. If it isn’t, we difference the data to make it stationary.
A simple first difference is $y_t = y_t ​− y_{t−1}$. This removes the trend so AR and MA can model what’s left.

## Example - Sunspot Activity
Before fitting ARIMA, we look at the ACF and PACF. The statsmodels sunspots series tracks sunspot activity over time, with clear cyclical swings.

```python
import matplotlib.pyplot as plt
from statsmodels.datasets import sunspots

# load data
data = sunspots.load_pandas().data['SUNACTIVITY']

# plot the time series
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.title('Sunspot activity over time')
plt.tight_layout()
plt.show()
```

![](/assets/images/arima/sunactivity.png){: width='90%'}

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_acf(data, ax=axes[0])

plot_pacf(data, ax=axes[1])

plt.tight_layout()
plt.show()
```

![](/assets/images/arima/acf-pacf.png){: width='95%'}

The `ACF` doesn’t die out after just a few lags. It fades out slowly and even flips sign, which means past shocks don’t fade right away.

The `PACF` jumps at lag 1 and lag 2, then settles down. That means today’s value is mostly driven by the last one or two observations.

**How to Choose $p$ and $q$:**     
> AR(p): PACF cuts off after lag $p$, while ACF tails off.    
MA(q): ACF cuts off after lag $q$, while PACF tails off.    
ARMA(p, q): Both ACF and PACF tail off.

The PACF spikes at lags 1 and 2. That puts 𝑝 at 2. The ACF tails off, which suggests there may also be an MA component. I'll start with **ARMA(2, 1)**.


```python
from statsmodels.tsa.arima.model import ARIMA

# define and fit the ARMA(2,1) model
model = ARIMA(data, order=(2, 0, 1)) # order=(p, d, q)
model_fit = model.fit()

# visualize the results
plt.figure(figsize=(10, 4))
plt.plot(data, label='Actual', alpha=0.8)
plt.plot(model_fit.fittedvalues, color='red', label='Fitted', alpha=0.7)
plt.title('Sunspot Activity: Actual vs. Fitted')
plt.legend()
plt.tight_layout()
plt.show()
```

![](/assets/images/arima/arima(2,0,1).png){: width='95%'}

Next, I'll check the residuals to see what the model left behind.
```python
# check model diagnostics on the residuals
model_fit.plot_diagnostics(figsize=(12, 8))
```
![](/assets/images/arima/plot-diagnostics.png){: width='95%'}

- The residuals are centered around zero, but they get more volatile later on. The size of the errors changes over time.
- The histogram and Q–Q plot look fine in the middle. But the tails bend away, so a few big errors show up.
- The residual ACF is mostly flat. There isn’t much time structure left for ARMA.

<br>
The ARIMA model did a decent job on the **mean**, but it missed the changing **variance**. *GARCH* is one way to model that.
<br>

ARIMA is still a good place to start. Once the time structure is gone, what’s left tells you whether you need a different ARMA order, a richer mean model like SARIMA or ARIMAX, or a different noise model like GARCH.


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