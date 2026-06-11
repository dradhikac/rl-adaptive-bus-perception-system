# ====================================
# RL STATE SPACE
# ====================================

state = {

    "traffic_signal": "RED",

    "vehicle_distance": "VERY_CLOSE",

    "risk_level": "HIGH",

    "blind_spot_left": "CAUTION",

    "blind_spot_right": "UNSAFE",

    "fusion_confidence": 0.91,

    "camera_reliability": 1.0
}

# ====================================
# ACTION SPACE
# ====================================

actions = [

    "MOVE",

    "SLOW_DOWN",

    "STOP",

    "CHANGE_LEFT",

    "CHANGE_RIGHT"
]

# ====================================
# DISPLAY
# ====================================

print("\n==========================")
print(" RL ENVIRONMENT ")
print("==========================")

print("\nSTATE:")

for key, value in state.items():

    print(key, ":", value)

print("\nACTIONS:")

for action in actions:

    print("-", action)

print("\n==========================")