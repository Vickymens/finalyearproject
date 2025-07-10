from datetime import datetime
from meteostat import Point, Daily
import pandas as pd

# Define location: Accra, Ghana
accra = Point(5.5571, -0.2012)

# Define time period (last 5 years)
start = datetime(2019, 1, 1)
end = datetime(2023, 12, 31)

# Get daily data
data = Daily(accra, start, end)
data = data.fetch()

# Filter for only temperature and precipitation
filtered_data = data[["tavg", "prcp"]]

# Save to CSV
filtered_data.to_csv("accra_weather_2019_2023.csv")
print("Historical data saved to 'accra_weather_2019_2023.csv'")
