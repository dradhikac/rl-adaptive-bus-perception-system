def calculate_risk(object_count):

    risk_score = 0

    # Pedestrian risk
    risk_score += object_count.get("person", 0) * 2

    # Vehicle risk
    risk_score += object_count.get("car", 0)

    # Bus/truck risk
    risk_score += object_count.get("bus", 0) * 3
    risk_score += object_count.get("truck", 0) * 3

    return risk_score


# Example
sample_objects = {
    "person": 5,
    "car": 8,
    "bus": 1
}

risk = calculate_risk(sample_objects)

print("Risk Score:", risk)