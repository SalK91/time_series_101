Time series data represents a sequence of data points collected over time. Unlike other types of data, time series data has a temporal aspect, where the order and timing of the data points matter. This makes time series analysis unique and requires specialized techniques and models to understand and predict future patterns or trends


## 1. Auto-correlation and Partial Auto-correlation
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

## 2. Unit Roots
 unit root is a feature of some stochastic processes (random processes). This stochastic process is a time series model where a single shock can have a persistent effect. This means that the impact of a single, random event can continue to influence the process indefinitely.The concept is closely tied to the idea of stationarity in a time series. A time series is said to be stationary if its statistical properties do not change over time. However, a time series with a unit root is non-stationary, as its mean and variance can change over time.

Unit roots are a problem in time series analysis because they can lead to non-stationarity. A unit root is a root of the characteristic equation of a time series model that is equal to 1. When a time series has a unit root, it means that the series is not stationary and its statistical properties, such as mean and variance, are not constant over time.

Non-stationary time series can be problematic for several reasons:

1. Difficulty in modeling: Non-stationary time series violate the assumptions of many statistical models, making it challenging to accurately model and forecast future values. Models like ARIMA (AutoRegressive Integrated Moving Average) assume stationarity, so non-stationary data can lead to unreliable predictions.

2. Spurious regression: Non-stationary time series can result in spurious regression, where two unrelated variables appear to be strongly correlated. This can lead to misleading conclusions and inaccurate interpretations of the relationship between variables.

3. Inefficient parameter estimation: Non-stationary time series can lead to inefficient parameter estimation. The estimates of model parameters may have large standard errors, reducing the precision and reliability of the estimated coefficients.

To address the issue of unit roots and non-stationarity, techniques like differencing or transforming the data can be used to make the time series stationary. Differencing involves taking the difference between consecutive observations to remove the trend or seasonality in the data. Transformations like logarithmic or power transformations can also be applied to stabilize the variance of the series.

It is important to identify and address unit roots in time series analysis to ensure reliable and accurate modeling and forecasting.

## 3. Dickey Fuller Test & Augmented Dickey Fuller Test
The Dickey-Fuller Test and the Augmented Dickey-Fuller Test are statistical tests used to determine if a time series data set is stationary or not. Stationarity is an important concept in time series analysis, as it assumes that the statistical properties of the data, such as mean and variance, remain constant over time.

## 4. ARMA (AutoRegressive Moving Average) Model
The ARMA model is a popular time series model that combines both autoregressive (AR) and moving average (MA) components. It is used to forecast future values of a time series based on its past values.

The autoregressive (AR) component of the ARMA model represents the linear relationship between the current observation and a certain number of lagged observations. It assumes that the current value of the time series is a linear combination of its past values. The order of the autoregressive component, denoted by p, determines the number of lagged observations included in the model.

The moving average (MA) component of the ARMA model represents the linear relationship between the current observation and a certain number of lagged forecast errors. It assumes that the current value of the time series is a linear combination of the forecast errors from previous observations. The order of the moving average component, denoted by q, determines the number of lagged forecast errors included in the model.

The ARMA model can be represented by the following equation:

Y_t = c + ϕ_1 * Y_(t-1) + ϕ_2 * Y_(t-2) + ... + ϕ_p * Y_(t-p) + ε_t + θ_1 * ε_(t-1) + θ_2 * ε_(t-2) + ... + θ_q * ε_(t-q)

where:
- Y_t is the current value of the time series.
- c is a constant term.
- ϕ_1, ϕ_2, ..., ϕ_p are the autoregressive coefficients.
- ε_t is the current forecast error.
- θ_1, θ_2, ..., θ_q are the moving average coefficients.
- Y_(t-1), Y_(t-2), ..., Y_(t-p) are the lagged values of the time series.
- ε_(t-1), ε_(t-2), ..., ε_(t-q) are the lagged forecast errors.

The ARMA model is commonly used for time series forecasting and can be estimated using various methods, such as maximum likelihood estimation or least squares estimation.


# ARIMA Model
 ARIMA includes an integration term, denoted as the "I" in ARIMA, which accounts for non-stationarity in the data. Non-stationarity refers to a situation where the statistical properties of a time series, such as mean and variance, change over time. ARIMA models can handle non-stationary data by differencing the series to achieve stationarity.

In ARIMA models, the integration order (denoted as "d") specifies how many times differencing is required to achieve stationarity. This is a parameter that needs to be determined or estimated from the data. ARMA models do not involve this integration order parameter since they assume stationary data.

e.g y_t original series
First Order ARIMA will be: z_t = y_t+1 - y_t
Second Order ARIMA will be: k_t = z_t+1 - z_t

# SARIMA

SARIMA stands for Seasonal AutoRegressive Integrated Moving Average model. It is an extension of the ARIMA model that incorporates seasonality into the modeling process. SARIMA models are particularly useful when dealing with time series data that exhibit seasonal patterns.

Seasonality refers to regular patterns or fluctuations in a time series data that occur at fixed intervals within a year, such as daily, weekly, monthly, or quarterly.
Seasonality is often caused by external factors like weather, holidays, or economic cycles.
Seasonal patterns tend to repeat consistently over time.

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








                               

