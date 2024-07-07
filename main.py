import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

csv_file_path = '/Users/srijitaseth/accident_analyser/US_Accidents_March23.csv'  

chunksize = 10**6  
chunks = pd.read_csv(csv_file_path, chunksize=chunksize)

data = pd.concat(chunks, ignore_index=True)

if 'Start_Lat' not in data.columns or 'Start_Lng' not in data.columns:
    raise ValueError("Missing 'Start_Lat' or 'Start_Lng' columns in the CSV file")

data.dropna(subset=['Start_Lat', 'Start_Lng'], inplace=True)

# Convert 'Start_Time' to datetime
data['Start_Time'] = pd.to_datetime(data['Start_Time'], errors='coerce')
data['Hour'] = data['Start_Time'].dt.hour
data['Day_of_Week'] = data['Start_Time'].dt.day_name()

# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(data['Start_Lng'], data['Start_Lat'])]
geo_data = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")


fig, ax = plt.subplots(figsize=(12, 10))
geo_data.plot(ax=ax, color='red', alpha=0.5, markersize=10)
ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)
plt.title('Accident Hotspots')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

road_conditions = data[['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']].sum()
road_conditions.plot(kind='bar', figsize=(12, 6), color='teal')
plt.title('Accidents by Road Conditions')
plt.xlabel('Road Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Weather_Condition', palette='plasma')
plt.title('Accidents by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Hour', palette='magma')
plt.title('Accidents by Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Day_of_Week', palette='cividis')
plt.title('Accidents by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

data['Weather_Condition_Numeric'] = pd.factorize(data['Weather_Condition'])[0]
correlation_data = data[['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Weather_Condition_Numeric', 'Hour']].copy()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Factors')
plt.show()
