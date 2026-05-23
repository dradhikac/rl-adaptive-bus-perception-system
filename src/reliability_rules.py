def calculate_camera_reliability(weather, time_of_day):

    reliability = 1.0

    # Weather impact
    if weather == "rainy":
        reliability -= 0.3

    elif weather == "foggy":
        reliability -= 0.5

    elif weather == "cloudy":
        reliability -= 0.1

    # Time impact
    if time_of_day == "night":
        reliability -= 0.3

    elif time_of_day == "dawn/dusk":
        reliability -= 0.2

    # Keep reliability within range
    reliability = max(0.0, min(reliability, 1.0))

    return reliability


# Example test
score = calculate_camera_reliability("foggy", "night")

print("Camera Reliability Score:", score)