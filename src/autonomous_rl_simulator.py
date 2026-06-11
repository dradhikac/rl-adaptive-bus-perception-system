import random
import pandas as pd

# ====================================
# STATE VARIABLES
# ====================================

traffic_signals = [
    "RED",
    "GREEN"
]

vehicle_distances = [
    "FAR",
    "MEDIUM",
    "VERY_CLOSE"
]

risk_levels = [
    "LOW",
    "HIGH"
]

blind_spots = [
    "SAFE",
    "UNSAFE"
]

actions = [

    "MOVE",

    "SLOW_DOWN",

    "STOP",

    "CHANGE_LEFT",

    "CHANGE_RIGHT"
]

# ====================================
# Q TABLE
# ====================================

q_table = {}

# ====================================
# CREATE STATE
# ====================================

def create_state():

    return (

        random.choice(
            traffic_signals
        ),

        random.choice(
            vehicle_distances
        ),

        random.choice(
            risk_levels
        ),

        random.choice(
            blind_spots
        ),

        random.choice(
            blind_spots
        )
    )

# ====================================
# REWARD FUNCTION
# ====================================

def reward_function(
    state,
    action
):

    signal, distance, risk, left, right = state

    reward = 0

    # --------------------
    # Traffic Signal
    # --------------------

    if signal == "RED":

        if action == "STOP":
            reward += 100

        else:
            reward -= 200

    if signal == "GREEN":

        if action == "MOVE":
            reward += 80

    # --------------------
    # Distance
    # --------------------

    if distance == "VERY_CLOSE":

        if action == "STOP":
            reward += 100

        elif action == "MOVE":
            reward -= 150

    elif distance == "MEDIUM":

        if action == "SLOW_DOWN":
            reward += 50

    # --------------------
    # Risk
    # --------------------

    if risk == "HIGH":

        if action == "STOP":
            reward += 120

        elif action == "MOVE":
            reward -= 100

    # --------------------
    # Blind Spot
    # --------------------

    if left == "UNSAFE":

        if action == "CHANGE_LEFT":
            reward -= 300

    if right == "UNSAFE":

        if action == "CHANGE_RIGHT":
            reward -= 300

    if left == "SAFE":

        if action == "CHANGE_LEFT":
            reward += 50

    if right == "SAFE":

        if action == "CHANGE_RIGHT":
            reward += 50

    return reward

# ====================================
# INITIALIZE Q TABLE
# ====================================

for signal in traffic_signals:

    for distance in vehicle_distances:

        for risk in risk_levels:

            for left in blind_spots:

                for right in blind_spots:

                    state = (
                        signal,
                        distance,
                        risk,
                        left,
                        right
                    )

                    q_table[state] = {

                        action: 0.0

                        for action in actions
                    }

# ====================================
# TRAINING
# ====================================

alpha = 0.1

episodes = 5000

print("\nTraining RL Agent...\n")

for episode in range(episodes):

    state = create_state()

    action = random.choice(
        actions
    )

    reward = reward_function(
        state,
        action
    )

    old_q = q_table[state][action]

    new_q = old_q + alpha * (

        reward - old_q

    )

    q_table[state][action] = new_q

# ====================================
# TEST POLICY
# ====================================

test_state = (

    "RED",

    "VERY_CLOSE",

    "HIGH",

    "SAFE",

    "UNSAFE"
)

best_action = max(

    q_table[test_state],

    key=q_table[test_state].get
)

# ====================================
# REPORT
# ====================================

print("=" * 40)
print(" AUTONOMOUS RL SIMULATOR ")
print("=" * 40)

print("\nTest State:")

print(test_state)

print("\nQ Values:")

for action, value in q_table[
    test_state
].items():

    print(
        action,
        "->",
        round(value, 2)
    )

print("\nBest Action:")

print(best_action)

print("=" * 40)

# ====================================
# SAVE POLICY
# ====================================

policy_rows = []

for state in q_table:

    best_action = max(
        q_table[state],
        key=q_table[state].get
    )

    policy_rows.append({

        "traffic_signal": state[0],

        "vehicle_distance": state[1],

        "risk_level": state[2],

        "blind_left": state[3],

        "blind_right": state[4],

        "best_action": best_action
    })

df = pd.DataFrame(
    policy_rows
)

df.to_csv(
    "outputs/rl_policy.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "outputs/rl_policy.csv"
)