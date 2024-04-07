Time series data represents a sequence of data points collected over time. Unlike other types of data, time series data has a temporal aspect, where the order and timing of the data points matter. This makes time series analysis unique and requires specialized techniques and models to understand and predict future patterns or trends


# Auto-sorrelation and Partial Auto-correlation
Auto-correlation and partial auto-correlation are statistical measures used in time series analysis to understand the relationship between data points in a sequence.

Auto-correlation measures the similarity between a data point and its lagged versions. In other words, it quantifies the correlation between a data point and the previous data points in the sequence. It helps identify patterns and dependencies in the data over time. Auto-correlation is often visualized using a correlogram, which is a plot of the correlation coefficients against the lag.

Partial auto-correlation, on the other hand, measures the correlation between a data point and its lagged versions while controlling for the influence of intermediate data points. It helps identify the direct relationship between a data point and its lagged versions, excluding the indirect relationships mediated by other data points. Partial auto-correlation is also visualized using a correlogram.

Both auto-correlation and partial auto-correlation are useful in time series analysis for several reasons:

* Identifying seasonality: Auto-correlation can help detect repeating patterns or seasonality in the data. If there is a significant correlation at a specific lag, it suggests that the data exhibits a repeating pattern at that interval.

* Model selection: Auto-correlation and partial auto-correlation can guide the selection of appropriate models for time series forecasting. By analyzing the patterns in the correlogram, you can determine the order of autoregressive (AR) and moving average (MA) components in models like ARIMA (AutoRegressive Integrated Moving Average).

# Stationarity
• Mean and SD is constant
• No seasonality

How to check for stationary:
    1. Visually
    2. Global vs Local Test
    3. Augmented Dickey Fuller Test
    
How to make Time Series Stationary if it is not stationary?
* Difference with last value

# White Noise
• Mean =0
• St. Dev is constant w.r.t  to time
• Correlation between lags is 0

# Unit Roots
 unit root is a feature of some stochastic processes (random processes). This stochastic process is a time series model where a single shock can have a persistent effect. This means that the impact of a single, random event can continue to influence the process indefinitely.The concept is closely tied to the idea of stationarity in a time series. A time series is said to be stationary if its statistical properties do not change over time. However, a time series with a unit root is non-stationary, as its mean and variance can change over time.

Unit roots are a problem in time series analysis because they can lead to non-stationarity. A unit root is a root of the characteristic equation of a time series model that is equal to 1. When a time series has a unit root, it means that the series is not stationary and its statistical properties, such as mean and variance, are not constant over time.

Non-stationary time series can be problematic for several reasons:

1. Difficulty in modeling: Non-stationary time series violate the assumptions of many statistical models, making it challenging to accurately model and forecast future values. Models like ARIMA (AutoRegressive Integrated Moving Average) assume stationarity, so non-stationary data can lead to unreliable predictions.

2. Spurious regression: Non-stationary time series can result in spurious regression, where two unrelated variables appear to be strongly correlated. This can lead to misleading conclusions and inaccurate interpretations of the relationship between variables.

3. Inefficient parameter estimation: Non-stationary time series can lead to inefficient parameter estimation. The estimates of model parameters may have large standard errors, reducing the precision and reliability of the estimated coefficients.

To address the issue of unit roots and non-stationarity, techniques like differencing or transforming the data can be used to make the time series stationary. Differencing involves taking the difference between consecutive observations to remove the trend or seasonality in the data. Transformations like logarithmic or power transformations can also be applied to stabilize the variance of the series.

It is important to identify and address unit roots in time series analysis to ensure reliable and accurate modeling and forecasting.

# Dickey Fuller Test & Augmented Dickey Fuller Test
The Dickey-Fuller Test and the Augmented Dickey-Fuller Test are statistical tests used to determine if a time series data set is stationary or not. Stationarity is an important concept in time series analysis, as it assumes that the statistical properties of the data, such as mean and variance, remain constant over time.



