library(foreign)
library(tseries)
library(lubridate)
library(zoo)
library(ggplot2)
library(TTR)
library(forecast)

setwd("D:/projects/rtime")

# Read the ARFF file
data <- read.arff("dataset.txt")
head(data)


# Create time series objects for each stock
data$Date <- as.Date(data$Date, format = "%m/%d/%Y")
symbols<- unique(data$Symbol)
ts_list<- list()


for (symbol in symbols) {
  stock_data <- data[data$Symbol == symbol, ]
  stock_ts <- ts(stock_data$Adj_Close, frequency = 252, start = c(year(min(stock_data$Date)), 1))
  stock_ts <- na.approx(stock_ts)
  ts_list[[symbol]] <- stock_ts
}


# Checking for missing values in each time series
for (symbol in symbols) {
  missing_values <- sum(is.na(ts_list[[symbol]]))
  cat(symbol, "missing values:", missing_values, "\n")
}

# Impute missing values using linear interpolation
for (symbol in symbols) {
  ts_list[[symbol]] <- na.approx(ts_list[[symbol]])
}

# Verify that missing values are gone
for (symbol in symbols) {
  missing_values <- sum(is.na(ts_list[[symbol]]))
  cat(symbol, "missing values:", missing_values, "\n")
}

# Select a few stocks to plot (e.g., the first 3)
selected_symbols <- symbols[1:3]

for (symbol in selected_symbols) {
  stock_data <- data[data$Symbol == symbol, ]
  p<-ggplot(stock_data, aes(x = Date, y = Adj_Close)) +
    geom_line() +
    labs(title = paste("Adjusted Closing Price for", symbol), x = "Date", y = "Adjusted Close") 
  print(p)
}

# Decompose the time series for one stock (e.g., ADANIPORTS)
decomp <- decompose(ts_list[["ADANIPORTS"]], type = "additive") #type additive or multiplicative
plot(decomp)

# Perform stationarity tests for a few stocks
for (symbol in selected_symbols) {
  cat("Stationarity tests for", symbol, ":\n")
  print(adf.test(ts_list[[symbol]])) # Augmented Dickey-Fuller test
  print(kpss.test(ts_list[[symbol]])) # KPSS test
  cat("\n")
}

# Plot ACF and PACF individually for a few stocks
selected_symbols <- symbols[1:3]  # Select a few stocks for demonstration

for (symbol in selected_symbols) {
  # Plot ACF
  par(mar = c(4, 4, 3, 2) + 0.1)  # Adjust margins
  acf(ts_list[[symbol]], main = paste("ACF for", symbol), lag.max = 20)
  
  # Plot PACF
  par(mar = c(4, 4, 3, 2) + 0.1)  # Adjust margins
  pacf(ts_list[[symbol]], main = paste("PACF for", symbol), lag.max = 20)
}

par(mfrow = c(1,1)) #reset par.

# Example of differencing
diff_ts <- diff(ts_list[["ADANIPORTS"]])
plot(diff_ts)


# Select a stock for demonstration (e.g., "ADANIPORTS")
selected_stock <- "ADANIPORTS" 

# Fit an ARIMA model
arima_model <- auto.arima(ts_list[[selected_stock]])

# Generate forecasts
forecast_object <- forecast(arima_model, h = 10) # Forecast for the next 10 periods

# Extract forecast data
last_date <- max(data$Date[data$Symbol == selected_stock]) #get the last date from the stock data
forecast_dates <- seq(last_date + 1, by = "day", length.out = length(forecast_object$mean)) #create a sequence of dates

# Extract forecast data
forecast_data <- data.frame(Date = forecast_dates, Forecast = forecast_object$mean)

# Create a data frame with actual and forecasted values
plot_data <- data.frame(
  Date = c(data$Date[data$Symbol == selected_stock], forecast_data$Date),
  Value = c(ts_list[[selected_stock]], forecast_data$Forecast),
  Type = c(rep("Actual", length(ts_list[[selected_stock]])), rep("Forecast", length(forecast_data$Forecast)))
)

# Filter data to show only the last 20 days
last_20_days <- tail(plot_data, 20)

# Create the plot
ggplot(data = last_20_days, aes(x = Date, y = Value, color = Type)) +
  geom_line() +
  labs(title = paste("Forecasts for", selected_stock, "(Last 20 Days)"), x = "Date", y = "Adjusted Close")

# Now we move on to ETS (Error, Trend, Seasonality) and Naive/Seasonal Naive Method

# Split data into training and test sets
train_size <- floor(0.8 * length(ts_list[[selected_stock]]))
train_ts <- window(ts_list[[selected_stock]], start = start(ts_list[[selected_stock]]), end = time(ts_list[[selected_stock]])[train_size])
test_ts <- window(ts_list[[selected_stock]], start = time(ts_list[[selected_stock]])[train_size + 1])

# STL + ETS Model (stlf)
stlf_forecast <- stlf(train_ts, h = length(test_ts))

# Naive Method
naive_forecast <- naive(train_ts, h = length(test_ts))

# Seasonal Naive Method
snaive_forecast <- snaive(train_ts, h = length(test_ts))

## Plot the forecasts
plot(stlf_forecast, main = paste("STL + ETS and Naive Forecasts for", selected_stock), ylab = "Adjusted Close")
lines(naive_forecast$mean, col = "red")
lines(snaive_forecast$mean, col = "blue")
lines(test_ts, col = "green")
legend("topleft", legend = c("STL + ETS", "Naive", "Seasonal Naive", "Actual"), col = c("black", "red", "blue", "green"), lty = 1)

# Evaluate the models
stlf_accuracy <- accuracy(stlf_forecast, test_ts)
naive_accuracy <- accuracy(naive_forecast, test_ts)
snaive_accuracy <- accuracy(snaive_forecast, test_ts)

print("STL + ETS Accuracy:")
print(stlf_accuracy)
print("Naive Accuracy:")
print(naive_accuracy)
print("Seasonal Naive Accuracy:")
print(snaive_accuracy)

selected_stock <- "ADANIPORTS"

# Difference the time series
diff_ts <- diff(ts_list[[selected_stock]])

# Re-run stationarity tests
cat("Stationarity tests for differenced", selected_stock, ":\n")
print(adf.test(diff_ts))
print(kpss.test(diff_ts))

# Manual ARIMA parameter selection and cross-validation
tryCatch({
  arima_cv_manual <- tsCV(ts(diff_ts), forecastfunction = function(x, h) forecast(arima(x, order = c(1, 0, 1)), h = h), h = 10)
  rmse_arima_cv_manual <- sqrt(mean(arima_cv_manual^2, na.rm = TRUE))
  print(paste("ARIMA (1,0,1) CV RMSE (differenced):", rmse_arima_cv_manual))
}, error = function(e) {
  cat("Error with ARIMA(1,0,1) on differenced data:", e$message, "\n")
})

tryCatch({
  arima_cv_manual2 <- tsCV(ts(diff_ts), forecastfunction = function(x, h) forecast(arima(x, order = c(2, 0, 2)), h = h), h = 10)
  rmse_arima_cv_manual2 <- sqrt(mean(arima_cv_manual2^2, na.rm = TRUE))
  print(paste("ARIMA (2,0,2) CV RMSE (differenced):", rmse_arima_cv_manual2))
}, error = function(e) {
  cat("Error with ARIMA(2,0,2) on differenced data:", e$message, "\n")
})

tryCatch({
  arima_cv_manual3 <- tsCV(ts(diff_ts), forecastfunction = function(x, h) forecast(arima(x, order = c(0, 0, 0)), h = h), h = 10)
  rmse_arima_cv_manual3 <- sqrt(mean(arima_cv_manual3^2, na.rm = TRUE))
  print(paste("ARIMA (0,0,0) CV RMSE (differenced):", rmse_arima_cv_manual3))
}, error = function(e) {
  cat("Error with ARIMA(0,0,0) on differenced data:", e$message, "\n")
})

# Cross-validation on original time series (with auto.arima)
# tryCatch({
#   arima_cv_auto_original <- tsCV(ts_list[[selected_stock]], forecastfunction = function(x, h) forecast(auto.arima(x), h = h), h = 10)
#   rmse_arima_cv_auto_original <- sqrt(mean(arima_cv_auto_original^2, na.rm = TRUE))
#   print(paste("ARIMA (auto) CV RMSE (original):", rmse_arima_cv_auto_original))
# }, error = function(e) {
#   cat("Error with ARIMA (auto) on original data:", e$message, "\n")
# })

# Auto ARIMA is taking too long to converge as data is non stationary, 
# i think we should leave this out



# Cross-validation on original time series (with manual parameters)
tryCatch({
  arima_cv_manual_original <- tsCV(ts_list[[selected_stock]], forecastfunction = function(x, h) forecast(arima(x, order = c(1, 1, 1)), h = h), h = 10)
  rmse_arima_cv_manual_original <- sqrt(mean(arima_cv_manual_original^2, na.rm = TRUE))
  print(paste("ARIMA (1,1,1) CV RMSE (original):", rmse_arima_cv_manual_original))
}, error = function(e) {
  cat("Error with ARIMA (1,1,1) on original data:", e$message, "\n")
})

tryCatch({
  arima_cv_manual_original2 <- tsCV(ts_list[[selected_stock]], forecastfunction = function(x, h) forecast(arima(x, order = c(2, 1, 2)), h = h), h = 10)
  rmse_arima_cv_manual_original2 <- sqrt(mean(arima_cv_manual_original2^2, na.rm = TRUE))
  print(paste("ARIMA (2,1,2) CV RMSE (original):", rmse_arima_cv_manual_original2))
}, error = function(e) {
  cat("Error with ARIMA (2,1,2) on original data:", e$message, "\n")
})

tryCatch({
  arima_cv_manual_original3 <- tsCV(ts_list[[selected_stock]], forecastfunction = function(x, h) forecast(arima(x, order = c(0, 1, 0)), h = h), h = 10)
  rmse_arima_cv_manual_original3 <- sqrt(mean(arima_cv_manual_original3^2, na.rm = TRUE))
  print(paste("ARIMA (0,1,0) CV RMSE (original):", rmse_arima_cv_manual_original3))
}, error = function(e) {
  cat("Error with ARIMA(0,1,0) on original data:", e$message, "\n")
})

# Comparison of RMSE values
cat("Comparison:\n")
cat(paste("ARIMA (1,0,1) CV RMSE (differenced):", rmse_arima_cv_manual, "\n"))
cat(paste("ARIMA (2,0,2) CV RMSE (differenced):", rmse_arima_cv_manual2, "\n"))
cat(paste("ARIMA (0,0,0) CV RMSE (differenced):", rmse_arima_cv_manual3, "\n"))
#cat(paste("ARIMA (auto) CV RMSE (original):", rmse_arima_cv_auto_original, "\n"))
cat(paste("ARIMA (1,1,1) CV RMSE (original):", rmse_arima_cv_manual_original, "\n"))
cat(paste("ARIMA (2,1,2) CV RMSE (original):", rmse_arima_cv_manual_original2, "\n"))
cat(paste("ARIMA (0,1,0) CV RMSE (original):", rmse_arima_cv_manual_original3, "\n"))

# Create a data frame for plotting
rmse_data <- data.frame(
  Model = c("ARIMA (1,0,1)", "ARIMA (2,0,2)", "ARIMA (0,0,0)", 
            "ARIMA (1,1,1) Original", "ARIMA (2,1,2) Original", "ARIMA (0,1,0) Original"),
  RMSE = c(rmse_arima_cv_manual, rmse_arima_cv_manual2, rmse_arima_cv_manual3,
          rmse_arima_cv_manual_original, rmse_arima_cv_manual_original2, rmse_arima_cv_manual_original3)
)

# Create the bar plot
ggplot(rmse_data, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "RMSE Comparison of ARIMA Models", x = "Model", y = "RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels for readability

# Time series cross-validation for ETS
ets_cv <- tsCV(ts_list[[selected_stock]], forecastfunction = function(x, h) forecast(stlf(x, h = h), h = h), h = 10)
print(ets_cv)
# Calculate RMSE from cross-validation
rmse_ets_cv <- sqrt(mean(ets_cv^2, na.rm = TRUE))
print(paste("ETS CV RMSE:", rmse_ets_cv))






# Calculate average daily close price for each stock
avg_close_prices <- aggregate(Adj_Close ~ Date + Symbol, data = data, FUN = mean)

# Plot average daily close prices for multiple stocks
ggplot(avg_close_prices, aes(x = Date, y = Adj_Close, color = Symbol)) +
  geom_line() +
  labs(title = "Average Daily Close Prices for Multiple Stocks",
       x = "Date",
       y = "Average Adjusted Close Price") +theme_minimal()



# Bollinger Bands and other plots for a selected stock (Here: AXISBANK)
selected_stock <- "AXISBANK"
stock_data <- data[data$Symbol == selected_stock, ]

# Calculate Bollinger Bands
stock_data$Adj_Close <- na.approx(stock_data$Adj_Close) # Impute NAs
stock_data$MA20 <- SMA(stock_data$Adj_Close, n = 20)
stock_data$SD20 <- runSD(stock_data$Adj_Close, n = 20)
stock_data$Upper <- stock_data$MA20 + 2 * stock_data$SD20
stock_data$Lower <- stock_data$MA20 - 2 * stock_data$SD20

# Plot Bollinger Bands
ggplot(stock_data, aes(x = Date, y = Adj_Close)) +
  geom_line() +
  geom_line(aes(y = MA20), color = "blue") +
  geom_line(aes(y = Upper), color = "red", linetype = "dashed") +
  geom_line(aes(y = Lower), color = "green", linetype = "dashed") +
  labs(title = paste("Bollinger Bands for", selected_stock),
       x = "Date",
       y = "Adjusted Close Price") +
  theme_minimal()
# Plot Volume
ggplot(stock_data, aes(x = Date, y = Volume)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = paste("Volume for", selected_stock),
       x = "Date",
       y = "Volume") +
  theme_minimal()

# Plot Moving Average (e.g., 50-day)
stock_data$MA50 <- SMA(stock_data$Adj_Close, n = 50)
ggplot(stock_data, aes(x = Date, y = Adj_Close)) +
  geom_line() +
  geom_line(aes(y = MA50), color = "purple") +
  labs(title = paste("50-Day Moving Average for", selected_stock),
       x = "Date",
       y = "Adjusted Close Price") +
  theme_minimal()

# Plotting multiple stock forecasts using Seasonal Naive
symbols <- unique(data$Symbol)
forecasts_snaive <- list()

for (symbol in symbols) {
  stock_ts <- ts(data$Adj_Close[data$Symbol == symbol], frequency = 252)
  stock_ts <- na.approx(stock_ts) #impute NA's
  forecasts_snaive[[symbol]] <- snaive(stock_ts, h = 10) #Forecast for the next 10 days
}

# Plot forecasts for a few selected stocks
selected_symbols <- symbols[1:3]




for (symbol in selected_symbols) {
  stock_data <- data[data$Symbol == symbol, ]
  stock_ts <- ts(stock_data$Adj_Close, frequency = 252)
  stock_ts <- na.approx(stock_ts)
  
  # Split the time series into training and test sets (last 20 values)
  train_size <- length(stock_ts) - 20
  train_ts <- window(stock_ts, start = start(stock_ts), end = time(stock_ts)[train_size])
  test_ts <- window(stock_ts, start = time(stock_ts)[train_size + 1])
  
  # Fit a Seasonal Naive model
  forecast_snaive <- snaive(train_ts, h = 20)
  
  # Get the dates for the actual and forecasted values
  actual_dates <- tail(stock_data$Date, 20)
  forecast_dates <- actual_dates
  
  # Create a data frame for plotting
  plot_data <- data.frame(
    Date = c(actual_dates, forecast_dates),
    Value = c(as.vector(test_ts), as.vector(forecast_snaive$mean)),
    Type = c(rep("Actual", 20), rep("Forecast", 20))
  )
  
  # Create the plot
  p<-ggplot(plot_data, aes(x = Date, y = Value, color = Type)) +
    geom_line() +
    labs(title = paste("Seasonal Naive Forecast for Last 20 Values of", symbol),
         x = "Date",
         y = "Adjusted Close") +
    theme_minimal()
  
  print(p)
}
