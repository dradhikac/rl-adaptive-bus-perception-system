import json
import pandas as pd

def calculate_reliability(weather, timeofday):

    reliability = 1.0

    # Weather penalties
    weather_penalty = {
        "clear": 0.0,
        "partly cloudy": 0.1,
        "overcast": 0.2,
        "rainy": 0.4,
        "foggy": 0.5,
        "snowy": 0.5
    }

    reliability -= weather_penalty.get(weather, 0.2)

    # Time penalties
    if timeofday == "night":
        reliability -= 0.3

    elif timeofday == "dawn/dusk":
        reliability -= 0.15

    reliability = max(0.0, min(reliability, 1.0))

    return round(reliability, 2)


# Load labels
label_path = "datasets/bdd100k/labels/bdd100k_labels_images_train.json"

with open(label_path, "r") as file:
    data = json.load(file)

# Generate reliability for first 10 entries
for item in data[:10]:

    attributes = item.get("attributes", {})

    weather = attributes.get("weather", "unknown")
    timeofday = attributes.get("timeofday", "unknown")

    score = calculate_reliability(weather, timeofday)

    print("--------------------------------")
    print("Weather:", weather)
    print("Time:", timeofday)
    print("Reliability Score:", score)
    
    results = []

for item in data[:100]:

    attributes = item.get("attributes", {})

    weather = attributes.get("weather", "unknown")
    timeofday = attributes.get("timeofday", "unknown")

    score = calculate_reliability(weather, timeofday)

    results.append({
        "weather": weather,
        "timeofday": timeofday,
        "reliability_score": score
    })

# Create DataFrame
df = pd.DataFrame(results)

# Save CSV
df.to_csv("outputs/reliability_scores.csv", index=False)

print("\nCSV file saved successfully.")