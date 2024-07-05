Time series data represents a sequence of data points collected over time. Unlike other types of data, time series data has a temporal aspect, where the order and timing of the data points matter. This makes time series analysis unique and requires specialized techniques and models to understand and predict future patterns or trends.

Time series data are characterized by three key patterns:
* Trend: This pattern indicates a long-term increase or decrease in the data.
* Seasonal: A seasonal pattern arises when a time series is influenced by seasonal factors, such as the time of year or day of the week. Seasonality occurs at a fixed and known period.
* Cyclic: A cyclic pattern appears when the data show rises and falls that do not occur at a fixed frequency. These fluctuations are usually driven by economic conditions and are often linked to the "business cycle," typically lasting at least two years [0].
 
## 0. Data Preprocessing
Steps for pre-processing of time-series data:

1. Normalization - Normalize the data so the mean of the time-series is 0 and the standard deviation is 1. This can be achieved by subtracting the mean and dividing by the standard deviation of the time-series.

2. Remove Trend - Remove the trend by taking the first difference of the time-series. This helps to stabilize the mean of the time-series and remove long-term trends.

3. Remove Changing Volatility - If the data exhibits changing volatility, this can be addressed by computing the yearly standard deviation and dividing each data point by the corresponding year's standard deviation. This step helps to stabilize the variance of the time-series.

4. Remove Seasonal Effect - Compute the mean for all months across all years and subtract the data point by its month's average. This helps in removing any seasonal patterns that might be present in the data.

5. Handle Missing Values - Identify and handle any missing values in the time-series. Common methods include interpolation, forward filling, or using statistical methods to estimate and fill in the missing data.

6. Outlier Detection and Treatment - Detect and treat any outliers in the data, which might skew the results. Outliers can be treated by capping, flooring, or using more advanced statistical methods to adjust their impact.

7. Stationarity Check - Ensure the time-series is stationary, meaning its statistical properties like mean, variance, and autocorrelation are constant over time. Use techniques such as the Augmented Dickey-Fuller (ADF) test to check for stationarity. If the series is not stationary, further transformations such as differencing or logarithmic transformations might be necessary.

8. Lagged Features Creation - Create lagged features to capture the temporal dependencies in the data. This involves creating new features that represent previous time points of the series.

9. Feature Engineering - Engineer additional features that might help in modeling the time-series data, such as rolling statistics (mean, variance), time-based features (day of the week, month, quarter), and external factors (e.g., holidays, weather conditions).


## 1. Time Series Characteristics
In addition to  the standard descriptive statistical measures of central tendency (mean, median, mode) and variance, timeseries is defined by its temporal dependence. Temporal dependence is measured through auto-correlation and partial auto-correlation, which help identify the relationships between data points over time and are essential for understanding patterns and making accurate forecasts.
 
### Auto-correlation and Partial Auto-correlation
Auto-correlation and partial auto-correlation are statistical measures used in time series analysis to understand the relationship between data points in a sequence.

Auto-correlation measures the similarity between a data point and its lagged versions. In other words, it quantifies the correlation between a data point and the previous data points in the sequence. It helps identify patterns and dependencies in the data over time. Auto-correlation is often visualized using a correlogram, which is a plot of the correlation coefficients against the lag.

Partial auto-correlation, on the other hand, measures the correlation between a data point and its lagged versions while controlling for the influence of intermediate data points. It helps identify the direct relationship between a data point and its lagged versions, excluding the indirect relationships mediated by other data points. Partial auto-correlation is also visualized using a correlogram.

Both auto-correlation and partial auto-correlation are useful in time series analysis for several reasons:

* Identifying seasonality: Auto-correlation can help detect repeating patterns or seasonality in the data. If there is a significant correlation at a specific lag, it suggests that the data exhibits a repeating pattern at that interval.

* Model selection: Auto-correlation and partial auto-correlation can guide the selection of appropriate models for time series forecasting. By analyzing the patterns in the correlogram, you can determine the order of autoregressive (AR) and moving average (MA) components in models like ARIMA (AutoRegressive Integrated Moving Average).


# 2. Stationarity
* Mean and standard-deviation of the timeseries is constant
* No seasonality

How to check for stationary:

    1. Visually
    2. Global vs Local Test
    3. Augmented Dickey Fuller Test
    
How to make Time Series Stationary if it is not stationary?
* Differencing:  e.g First-order Differencing: Subtract the previous observation from the current observation. If the time series has seasonality, seasonal differencing can be applied.
* Transformations: Transformations like logarithm, square root, or Box-Cox can stabilize the variance.
* Decomposition: Decompose the time series into trend, seasonal, and residual components.
* Detrending e.g. Subtracting the Rolling Mean or Fitting and Removing a Linear Trend.



## White Noise
* Mean =0
* St. Dev is constant w.r.t  to time
* Correlation between lags is 0

## Unit Roots
 Unit root is a feature of some stochastic processes (random processes). This stochastic process is a time series model where a single shock can have a persistent effect. This means that the impact of a single, random event can continue to influence the process indefinitely.The concept is closely tied to the idea of stationarity in a time series. A time series is said to be stationary if its statistical properties do not change over time. However, a time series with a unit root is non-stationary, as its mean and variance can change over time.

Unit roots are a problem in time series analysis because they can lead to non-stationarity. A unit root is a root of the characteristic equation of a time series model that is equal to 1. When a time series has a unit root, it means that the series is not stationary and its statistical properties, such as mean and variance, are not constant over time.

Non-stationary time series can be problematic for several reasons:

1. Difficulty in modeling: Non-stationary time series violate the assumptions of many statistical models, making it challenging to accurately model and forecast future values. Models like ARIMA (AutoRegressive Integrated Moving Average) assume stationarity, so non-stationary data can lead to unreliable predictions.

2. Spurious regression: Non-stationary time series can result in spurious regression, where two unrelated variables appear to be strongly correlated. This can lead to misleading conclusions and inaccurate interpretations of the relationship between variables.

3. Inefficient parameter estimation: Non-stationary time series can lead to inefficient parameter estimation. The estimates of model parameters may have large standard errors, reducing the precision and reliability of the estimated coefficients.

To address the issue of unit roots and non-stationarity, techniques like differencing or transforming the data can be used to make the time series stationary. Differencing involves taking the difference between consecutive observations to remove the trend or seasonality in the data. Transformations like logarithmic or power transformations can also be applied to stabilize the variance of the series.

**It is important to identify and address unit roots in time series analysis to ensure reliable and accurate modeling and forecasting.**

## Dickey Fuller Test & Augmented Dickey Fuller Test
The Dickey-Fuller Test and the Augmented Dickey-Fuller Test are statistical tests used to determine if a time series data set is stationary or not. Stationarity is an important concept in time series analysis, as it assumes that the statistical properties of the data, such as mean and variance, remain constant over time.


# 3. Exponential Smoothing
Exponential smoothing is a time series forecasting technique that applies weighted averages to past observations, giving more weight to recent observations while exponentially decreasing the weight for older observations. This method is useful for making short-term forecasts and smoothing out irregularities in the data.

### Simple Exponential Smoothing:

The forecast for time $t+1$ is calculated as:

$$
l_{t+1} = \hat{y}_{t+1|t} = \alpha y_t + \alpha (1 - \alpha) y_{t-1} + \alpha (1 - \alpha)^2 y_{t-2} + \alpha (1 - \alpha)^3 y_{t-3} + \cdots 
$$

$$
l_{t+1} =  \alpha y_t + (1 - \alpha) l_{t} $$

where:

- $l_{t+1}/\hat{y}_{t+1|t} $ is the forecast for the next time period.
- $\alpha $ is the smoothing parameter (0 < $\alpha$ < 1).
- $y_t$ is the actual value at time $t$.
- $y_{t-1}$ is the actual value at time $t-1$.


# 3. ARMA (AutoRegressive Moving Average) Model
The ARMA model is a popular time series model that combines both autoregressive (AR) and moving average (MA) components. It is used to forecast future values of a time series based on its past values.

The autoregressive (AR) component of the ARMA model represents the linear relationship between the current observation and a certain number of lagged observations. It assumes that the current value of the time series is a linear combination of its past values. The order of the autoregressive component, denoted by p, determines the number of lagged observations included in the model.

The moving average (MA) component of the ARMA model represents the linear relationship between the current observation and a certain number of lagged forecast errors. It assumes that the current value of the time series is a linear combination of the forecast errors from previous observations. The order of the moving average component, denoted by q, determines the number of lagged forecast errors included in the model.

The ARMA model can be represented by the following equation:

$$Y_t = c + \phi_1  Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p  Y_{t-p} + \epsilon_t + \beta_1 \epsilon_{t-1} + \beta_2  \epsilon_{t-2} + ... + \beta_q  \epsilon_{t-q}$$



where:
- $Y_t$ is the current value of the time series.
- c is a constant term.
- $\phi_1, \phi_2, ..., \phi_p$ are the autoregressive coefficients.
- $ε_t$ is the current forecast error.
- $\beta_1, \beta_2, ..., \beta_q$ are the moving average coefficients.
- $Y_{t-1}, Y_{t-2}, ..., Y_{t-p}$ are the lagged values of the time series.
- $\epsilon_{t-1}, \epsilon_{t-2}, ..., \epsilon_{t-q}$ are the lagged forecast errors.

The ARMA model is commonly used for time series forecasting and can be estimated using various methods, such as maximum likelihood estimation or least squares estimation.


## ARIMA Model
 ARIMA includes an integration term, denoted as the "I" in ARIMA, which accounts for non-stationarity in the data. Non-stationarity refers to a situation where the statistical properties of a time series, such as mean and variance, change over time. ARIMA models can handle non-stationary data by differencing the series to achieve stationarity.

In ARIMA models, the integration order (denoted as "d") specifies how many times differencing is required to achieve stationarity. This is a parameter that needs to be determined or estimated from the data. ARMA models do not involve this integration order parameter since they assume stationary data.

e.g $y_t$ original series

* First Order ARIMA will be: $z_t = y_{t+1} - y_t$
* Second Order ARIMA will be: $k_t = z_{t+1} - z_t$

## SARIMA

SARIMA stands for Seasonal AutoRegressive Integrated Moving Average model. It is an extension of the ARIMA model that incorporates seasonality into the modeling process. SARIMA models are particularly useful when dealing with time series data that exhibit seasonal patterns.

Seasonality refers to regular patterns or fluctuations in a time series data that occur at fixed intervals within a year, such as daily, weekly, monthly, or quarterly. Seasonality is often caused by external factors like weather, holidays, or economic cycles. Seasonal patterns tend to repeat consistently over time.

How to address seasonality in time-series models:
* Identify Seasonality: Begin by examining the time series data to detect any patterns that repeat at regular intervals. Seasonality refers to variations in the data that occur at specific time intervals, such as daily, weekly, monthly, or quarterly patterns.
Here's how you can identify a seasonal trend using the ACF:

1. Periodic Peaks: Look for peaks in the ACF plot at regular intervals, corresponding to the seasonal lag. For example, if you're analyzing monthly data and suspect a yearly seasonality, you would expect peaks at lags 12, 24, 36, and so on. Similarly, for quarterly data, peaks would occur at lags 4, 8, 12, and so forth.

2. Significant Peaks: Pay attention to the magnitude of the autocorrelation coefficients at seasonal lags. If the peaks at seasonal lags are significantly higher compared to other lags, it suggests a strong seasonal pattern in the data.

3. Repetitive Patterns: Check for repetitive patterns in the ACF plot that align with the seasonal frequency of the data. Seasonal trends often exhibit periodicity, resulting in a repeating pattern of autocorrelation coefficients at seasonal lags.

4. Alternating Positive and Negative Correlations: In some cases, you may observe alternating positive and negative autocorrelation coefficients at seasonal lags, indicating a seasonal pattern in the data.

5. Partial Autocorrelation Function (PACF): Additionally, you can complement your analysis with the Partial Autocorrelation Function (PACF), which helps identify the direct influence of a lag on the current observation, excluding the indirect effects through shorter lags. Significant spikes in PACF at seasonal lags provide further evidence of seasonality in the data.

By carefully examining the ACF plot for these indicators, you can infer the presence of a seasonal trend in the time series data. This insight is crucial for selecting appropriate forecasting models and designing interventions to address seasonality in the data.

* Remove Seasonality: Once the seasonal component has been identified, it needs to be removed from the original data. This can be achieved by differencing the data at seasonal intervals. For example, if the data exhibits monthly seasonality, you can difference the data by subtracting each observation from the observation from the same month in the previous year.

* Fit ARIMA Model: After removing seasonality, fit an ARIMA model to the deseasonalized data. ARIMA models are effective for modeling the remaining non-seasonal components of the time series, including trend and random noise.


Cycles, on the other hand, refer to fluctuations in a time series that are not of fixed frequency or period.
Cycles are typically longer-term patterns, often spanning several years, and are not as precisely defined as seasonal patterns.
Cycles can be influenced by economic factors, business cycles, or other structural changes in the data.
In summary, while both seasonality and cycles involve patterns of variation in time series data, seasonality repeats at fixed intervals within a year, whereas cycles represent longer-term fluctuations that may not have fixed periodicity.


# 4. ARCH Model
ARCH (Autoregressive Conditional Heteroskedasticity) models are a class of models used in econometrics and financial econometrics to analyze time series data, particularly in the context of volatility clustering. These models are designed to capture the time-varying volatility or heteroskedasticity in financial time series data, where the volatility of the series may change over time.

> In statistics, a sequence of random variables is homoscedastic if all its random variables have the same finite variance; this is also known as homogeneity of variance. The complementary notion is called heteroscedasticity, also known as heterogeneity of variance [1]

The basic idea behind ARCH models is that the variance of a time series can be modeled as a function of its own past values, along with possibly some exogenous variables. In other words, the variance at any given time is conditional on the past observations of the series. 

### ARCH (1) Model Derivation
It posits that the observed value at time t can be decomposed into an average component $\mu$ and a noise term $𝑎(𝑡)$

$$ 𝑦(𝑡)= \mu + 𝑎(𝑡) $$ 

The noise term $𝑎(𝑡)$ is further defined as the product of a time dependent volatility term $\sigma(t)$ and a stochastic noise component $\epsilon (t)$

$$𝑎(𝑡)=\sigma(𝑡) \epsilon(𝑡)$$


In the ARCH model, the volatility term $\sigma(𝑡)$ is modeled as a function of past values of the noise term $a(t)$. Specifically, the ARCH(1) model (the simplest form) defines $\sigma (t)$ as:

$$\sigma(t) = \sqrt{\alpha_0 + \alpha_1 * a^2 _{t-1}}$$ 
 
Where $\alpha_0, \alpha_1$ are the parameters of the model and $𝑎_{𝑡−1}$ is the volatility at the last timestep.

### ARCH (p) Model
The ARCH(1) model can be generalized to an ARCH(p) model, where the volatility term depends on the past $p$ values of the noise term $a(t)$:

$$ \text{ARCH}(p): \quad \sigma^2_t = \alpha_0 + \alpha_1 a^2_{t-1} + \alpha_2 a^2_{t-2} + \ldots + \alpha_p a^2_{t-p} $$

where:
- $ \sigma^2_t $ is the conditional variance of the time series at time t.
- $a_t$ is the error term at time t.
- $\alpha_0, \alpha_1, \alpha_2, \ldots, \alpha_p$ are parameters to be estimated.
- $p$ is the order of the ARCH model, indicating how many past squared residuals are included in the model.


The final formulation of the ARCH(p) model is:

1. The observed value:
   $$y(t) = \mu + a(t)$$

2. The noise term:
   $$a(t) = \sigma(t) \epsilon(t)$$

3. The volatility term:
   $$\sigma^2(t) = \alpha_0 + \sum_{i=1}^{p} \alpha_i a^2(t-i)$$
   where $\epsilon(t)$ is white noise with zero mean and unit variance ($ \epsilon(t) \sim N(0, 1) $).


To estimate the parameters, one typically uses maximum likelihood estimation (MLE) or other estimation techniques. Once the parameters are estimated, the model can be used to forecast the conditional variance of the time series into the future.

## GARCH Model
GARCH model is extenstion of ARCH Model. It models time series as a function of previous states value as well volatality. GARCH compared to ARCH takes volatality of time-series into account.

The volatility term $\sigma^2(t)$ in the GARCH(1, 1) model is defined as:

 $$\sigma^2(t) = \alpha_0 + \alpha_1 a^2(t-1) + \beta_1 \sigma^2(t-1)$$

The ARCH and GARCH models are crucial in modeling time series data with time-varying volatility. ARCH models capture conditional heteroskedasticity by modeling volatility as a function of past squared errors, while GARCH models extend this to include past volatility terms, providing a more comprehensive framework for volatility modeling.


# Review - ARMA & GARCH
* AR/ARMA Models: Best suited for stationary time series data, where statistical properties like mean and variance are constant over time. Useful for short-term forecasting, ARMA models combine both autoregressive (AR) and moving average (MA) components to capture the dynamics influenced by past values and past forecast errors.

* AR Models: Used when the primary relationship in the data is between the current value and its own past values. Suitable for time series where residuals show no significant autocorrelation pattern, indicating that past values alone sufficiently explain the current observations.

* ARMA Models: Employed when both past values and past forecast errors significantly influence the current value. This combination provides a more comprehensive model for capturing complex dynamics in time series data.

* ARCH Models: Best suited for time series data with volatility clustering but lacking long-term persistence. ARCH models capture bursts of high and low volatility effectively by modeling changing variance over time based on past errors.

* GARCH Models: Extend ARCH models by incorporating past variances, allowing them to handle more persistent volatility. GARCH models are better at capturing long-term dependencies in financial time series data, making them suitable for series with sustained periods of high or low volatility.


![ARMA - GARCH Review](images/Figure_ARMA_GARCH_Review.png)

# 5. Vector Autoregression

Vector autoregressive models are used for multivariate time series. It is used capture the linear interdependencies among multiple time series. The VAR model generalizes the univariate autoregressive (AR) model by allowing for more than one evolving variable. The structure is that each variable is a linear function of past lags of itself and past lags of the other variables [2].


### Three-Variable VAR(1) Model

Consider a three-variable time series $ \mathbf{y}_t = (y_{1t}, y_{2t}, y_{3t})' $. The VAR(1) model for this time series can be written as:

$$
y_{t,1} = \alpha_{0,1} +  \alpha_{11}y_{t-1,1} +  \alpha_{12}y_{t-1,2} +  \alpha_{13}y_{t-1,3} + \epsilon_{t,1}
$$

$$
y_{t,2} = \alpha_{0,2} +  \alpha_{21}y_{t-1,1} +  \alpha_{22}y_{t-1,2} +  \alpha_{23}y_{t-1,3} + \epsilon_{t,1}
$$

$$
y_{t,3} = \alpha_{0,3} +  \alpha_{31}y_{t-1,1} +  \alpha_{32}y_{t-1,2} +  \alpha_{33}y_{t-1,3} + \epsilon_{t,1}
$$

i.e. each variable is a linear function of the lag 1 values for all variables in the set. This in matrix form can be expressed as:
$$
\mathbf{Y}_t = \mathbf{\alpha_0} + \mathbf{A}_1 \mathbf{Y}_{t-1} + \mathbf{\epsilon}_t
$$


where:
- $ \mathbf{Y}_t$ is a 3x1 vector of time series variables at time $ t $.
- $ \mathbf{\alpha_0} $ is a 3x1 vector of intercept terms.
- $ \mathbf{A}_1 $ is  a 3x1 coefficient matrix.
- $ \mathbf{\epsilon}_t $ is a  3x1 vector of error terms (white noise).

# 6. Granger Causality
> The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another [3]. It assesses whether past values of one variable $X$ provide statistically significant information about future values of another variable $Y$.

1. The method involves constructing regression models:

   - An AR model where $Y $is regressed on its own past values.
   - A VAR model where $Y$ is regressed on its own past values and past values of $X$.

2. Hypothesis Testing - The primary statistical test involves a null hypothesis that $X$ does not Granger-cause $Y$. If adding past values of $X$ improves the prediction of $Y$ significantly, the null hypothesis is rejected, indicating that $X$ Granger-causes $Y$.

Granger causality does not imply true causality in the philosophical sense; it only indicates predictive causality based on the given data.

# 7. Model Selection
When analyzing time series data, selecting the appropriate model e.g. AR vs ARMA and model's order is crucial for making accurate predictions. Model selecition methods include:

###  Akaike Information Criterion  (AIC)
$$AIC=2k−2ln(L)$$
where,
* $k$ - Number of parameters in the model
* $L$ - Likelihood function for the model

### Bayesian Information Criterion (BIC)
$$BIC=kln(n)−2ln(L)$$
where,
* $k$ - Number of parameters in the model.
* $n$ - Number of data points.
* $L$ - Likelihood function for the model.

### Cross Validation
Divide the time series into training and testing sets. A common method is time series cross-validation, where the data is split into multiple training and validation sets in a rolling or expanding window manner. Use metrics such as Mean Squared Error (MSE) or Mean Absolute Error (MAE) to evaluate performance to choose the best model.

# 8. Anomaly Detection
Anomalies refer to data values or events that deviate significantly from the normal trend. Detecting and correcting anomalies is crucial before any analysis of data, as anomalies can lead to incorrect results and conclusions. However, the time dependence and often non-stationary nature of time series data make anomaly detection particularly complex.

Anomaly detection methods:

1. Z-Score Method - calculates the standard score of each data point to identify anomalies.
2. Moving Average with Standard Deviation -  deviation of the point from the rolling moving average and standard deviation.
3. Isolation Forest - an ensemble method designed for anomaly detection.
4. One-Class SVM - One-Class Support Vector Machines can be used for anomaly detection.
5. Kmeans clustering - unsupervised clustering can help identify anomalies.
6. STL decomposition - “Seasonal and Trend decomposition using Loess”  decomposes time series into it seasonal, trend, and residue components. Residue component can be used to identify anomalies.
7. Detection using Forecasting -  In forecasting we predict each point based on past data points using a forecasting method such as ARMA model. The deviation of actual value from the prediction can be used to identify anomalies.

# 9. Bayesian Time Series
Bayesian time series modelling incorporates Bayesian statistical methods to model and analyze time-dependent data. It leverages prior distributions along with observed data to make inferences about the underlying processes generating the time series. Bayesian methods provide a coherent framework for incorporating prior knowledge and quantifying uncertainty in parameter estimates.

# 10. Recurrent Neural Networks

## 11. LSTM 
## Read this boo
https://otexts.com/fpp3/
Resume for chapter 8 - https://otexts.com/fpp3/expsmooth.html

chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/steel/steel_homepage/bayesiantsrev.pdf
State spaxe model

chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf

https://www.pymc.io/projects/docs/en/stable/learn.html
https://facebook.github.io/prophet/

## References
0. https://otexts.com/fpp3/ 
1. https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity
2. https://online.stat.psu.edu/stat510/lesson/11/11.2
3. https://en.wikipedia.org/wiki/Granger_causality




                               

