# ====================================
# REWARD FUNCTION
# ====================================

def calculate_reward(

    state,

    action

):

    reward = 0

    signal = state["traffic_signal"]

    distance = state["vehicle_distance"]

    risk = state["risk_level"]

    left = state["blind_spot_left"]

    right = state["blind_spot_right"]

    # ================================
    # TRAFFIC SIGNAL
    # ================================

    if signal == "RED":

        if action == "STOP":
            reward += 100

        else:
            reward -= 300

    elif signal == "GREEN":

        if action == "MOVE":
            reward += 80

    # ================================
    # DISTANCE
    # ================================

    if distance == "VERY_CLOSE":

        if action == "STOP":
            reward += 100

        elif action == "MOVE":
            reward -= 250

    # ================================
    # RISK
    # ================================

    if risk == "HIGH":

        if action == "STOP":
            reward += 120

        elif action == "MOVE":
            reward -= 200

    # ================================
    # BLIND SPOT
    # ================================

    if left == "UNSAFE":

        if action == "CHANGE_LEFT":
            reward -= 400

    if right == "UNSAFE":

        if action == "CHANGE_RIGHT":
            reward -= 400

    # ================================
    # SAFE LANE CHANGES
    # ================================

    if left == "SAFE":

        if action == "CHANGE_LEFT":
            reward += 50

    if right == "SAFE":

        if action == "CHANGE_RIGHT":
            reward += 50

    return reward

# ====================================
# TEST CASE
# ====================================

state = {

    "traffic_signal": "RED",

    "vehicle_distance": "VERY_CLOSE",

    "risk_level": "HIGH",

    "blind_spot_left": "CAUTION",

    "blind_spot_right": "UNSAFE"
}

actions = [

    "MOVE",

    "SLOW_DOWN",

    "STOP",

    "CHANGE_LEFT",

    "CHANGE_RIGHT"
]

print("\n==========================")
print(" REWARD FUNCTION TEST ")
print("==========================")

for action in actions:

    reward = calculate_reward(
        state,
        action
    )

    print(
        action,
        "->",
        reward
    )