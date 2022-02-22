import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
import seaborn as sns

df = pd.read_csv('avocado.csv')

year, month, day = [], [], []
for i in df["Date"]:
    ts = pd.to_datetime(i)
    year.append(ts.year)
    month.append(ts.month)
    day.append(ts.day)

df["year"] = year
df["month"] = month
df["day"] = day
df["Date"] = pd.to_datetime(df["Date"])

df = df[["Date", "AveragePrice", "region", "year", "month", "day", "type"]]

plt.figure(figsize=(12, 5))
plt.title("Fiyat Dağılım Grafiği")
sns.distplot(df["AveragePrice"])
plt.show()

date_grouped = df.groupby(["Date"]).mean()[["AveragePrice"]]
plt.figure(figsize=(12, 5))
plt.plot(date_grouped.index, date_grouped["AveragePrice"].values)
plt.title('Ortalama fiyatın değişim grafiği')
plt.show()

year_grouped = df.groupby(["year"]).mean()[["AveragePrice"]]
month_grouped = df.groupby(["month"]).mean()[["AveragePrice"]]
day_grouped = df.groupby(["day"]).mean()[["AveragePrice"]]

plt.figure(figsize=(12, 5))
plt.plot(year_grouped.index, year_grouped["AveragePrice"].values)
plt.title('Ortalama fiyatın yıllara göre değişim grafiği')
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
plt.plot(month_grouped.index, month_grouped["AveragePrice"].values)
ax.xaxis.set(ticks=range(0, 13))  # Manually set x-ticks
plt.title('Ortalama fiyatın aylara göre değişim grafiği')
plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
plt.plot(day_grouped.index, day_grouped["AveragePrice"].values)
ax.xaxis.set(ticks=range(0, 32))  # Manually set x-ticks
plt.title('Ortalama fiyatın günlere göre değişim grafiği')
plt.show()

data = df[["Date", "AveragePrice"]]

data = data.rename(columns={
    'Date': 'ds',
    'AveragePrice': 'y'
})

ax = data.set_index('ds').plot(figsize=(20, 12))
ax.set_ylabel('Monthly Average Price of Avocado')
ax.set_xlabel('Date')
plt.show()

my_model = Prophet()
my_model.fit(data)

future_dates = my_model.make_future_dataframe(periods=900)
forecast = my_model.predict(future_dates)

fig2 = my_model.plot_components(forecast)

forecast_df = forecast[["ds", "yhat"]]

mask = (forecast_df['ds'] > "2018-03-24") & (forecast_df['ds'] <= "2020-09-10")
forecastedvalues = forecast_df.loc[mask]

mask = (forecast_df['ds'] > "2015-01-04") & (forecast_df['ds'] <= "2018-03-25")
forecast_df = forecast_df.loc[mask]

fig, ax1 = plt.subplots(figsize=(16, 8))
ax1.plot(forecast_df.set_index('ds'), color='b')
ax1.plot(forecastedvalues.set_index('ds'), color='r')
ax1.set_ylabel('Ortalama Fiyat')
ax1.set_xlabel('Tarih')
plt.show()
