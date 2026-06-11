import pandas as pd

# ====================================
# STATE
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
# ACTIONS
# ====================================

actions = [

    "MOVE",

    "SLOW_DOWN",

    "STOP",

    "CHANGE_LEFT",

    "CHANGE_RIGHT"
]

# ====================================
# REWARD FUNCTION
# ====================================

def calculate_reward(state, action):

    reward = 0

    if state["traffic_signal"] == "RED":

        if action == "STOP":
            reward += 100
        else:
            reward -= 200

    if state["vehicle_distance"] == "VERY_CLOSE":

        if action == "STOP":
            reward += 100
        elif action == "MOVE":
            reward -= 150

    if state["risk_level"] == "HIGH":

        if action == "STOP":
            reward += 120
        elif action == "MOVE":
            reward -= 100

    if state["blind_spot_right"] == "UNSAFE":

        if action == "CHANGE_RIGHT":
            reward -= 300

    if state["blind_spot_left"] == "SAFE":

        if action == "CHANGE_LEFT":
            reward += 50

    return reward

# ====================================
# EVALUATE ACTIONS
# ====================================

results = []

for action in actions:

    reward = calculate_reward(
        state,
        action
    )

    results.append({

        "action": action,

        "reward": reward
    })

df = pd.DataFrame(results)

print("\n==========================")
print(" ADAPTIVE RL AGENT ")
print("==========================")

print(df)

best_action = df.loc[
    df["reward"].idxmax()
]["action"]

print("\nBest Action:")
print(best_action)