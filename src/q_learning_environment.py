import random
import pandas as pd

# ====================================
# STATES
# ====================================

states = [

    "SAFE",

    "CAUTION",

    "DANGER"
]

# ====================================
# ACTIONS
# ====================================

actions = [

    "MOVE",

    "SLOW_DOWN",

    "STOP"
]

# ====================================
# Q TABLE
# ====================================

q_table = pd.DataFrame(
    0.0,
    index=states,
    columns=actions
)

# ====================================
# REWARD FUNCTION
# ====================================

def get_reward(

    state,

    action

):

    if state == "SAFE":

        if action == "MOVE":
            return 100

        if action == "SLOW_DOWN":
            return 20

        return -50

    if state == "CAUTION":

        if action == "SLOW_DOWN":
            return 100

        if action == "STOP":
            return 40

        return -100

    if state == "DANGER":

        if action == "STOP":
            return 150

        return -200

# ====================================
# TRAINING
# ====================================

alpha = 0.1

episodes = 1000

for episode in range(episodes):

    state = random.choice(
        states
    )

    action = random.choice(
        actions
    )

    reward = get_reward(
        state,
        action
    )

    old_value = q_table.loc[
        state,
        action
    ]

    new_value = (

        old_value +

        alpha *

        (reward - old_value)

    )

    q_table.loc[
        state,
        action
    ] = new_value

# ====================================
# RESULTS
# ====================================

print("\n==========================")
print(" TRAINED Q TABLE ")
print("==========================")

print(q_table)

print("\n==========================")
print(" BEST ACTIONS ")
print("==========================")

for state in states:

    best_action = q_table.loc[
        state
    ].idxmax()

    print(

        state,

        "->",

        best_action

    )