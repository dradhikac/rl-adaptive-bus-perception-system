import json
from collections import Counter

# Label file path
label_path = "datasets/bdd100k/labels/bdd100k_labels_images_train.json"

# Open JSON file
with open(label_path, "r") as file:
    data = json.load(file)

# Total entries
print("Total Label Entries:", len(data))

# First sample
first_entry = data[0]

print("\nFirst Entry:")
print(first_entry)

# Extract weather and time information

weather_list = []
time_list = []

for item in data:

    attributes = item.get("attributes", {})

    weather = attributes.get("weather", "unknown")
    timeofday = attributes.get("timeofday", "unknown")

    weather_list.append(weather)
    time_list.append(timeofday)

# Display first 10 weather labels
print("\nWeather Samples:")
print(weather_list[:10])

print("\nTime of Day Samples:")
print(time_list[:10])

# Count weather frequency
weather_counter = Counter(weather_list)

print("\nWeather Distribution:")
print(weather_counter)