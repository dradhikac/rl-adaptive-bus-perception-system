import pandas as pd

# ====================================
# LOAD POLICY
# ====================================

df = pd.read_csv(
    "outputs/rl_policy.csv"
)

print("\nLoading RL Policy...")

# ====================================
# DISPLAY SAMPLE RULES
# ====================================

print("\n================================")
print(" AUTONOMOUS DRIVING POLICY ")
print("================================")

for i in range(min(20, len(df))):

    row = df.iloc[i]

    print(
        f"\nRule {i+1}"
    )

    print(

        f"IF Signal={row['traffic_signal']} "

        f"AND Distance={row['vehicle_distance']} "

        f"AND Risk={row['risk_level']} "

        f"AND Left={row['blind_left']} "

        f"AND Right={row['blind_right']}"

    )

    print(
        f"THEN {row['best_action']}"
    )

# ====================================
# POLICY SUMMARY
# ====================================

print("\n================================")
print(" POLICY STATISTICS ")
print("================================")

print(

    df["best_action"]
    .value_counts()

)

# ====================================
# SAVE POLICY REPORT
# ====================================

with open(
    "outputs/autonomous_policy.txt",
    "w"
) as f:

    f.write(
        "AUTONOMOUS DRIVING POLICY\n\n"
    )

    for _, row in df.iterrows():

        f.write(

            f"IF "

            f"{row['traffic_signal']} "

            f"{row['vehicle_distance']} "

            f"{row['risk_level']} "

            f"{row['blind_left']} "

            f"{row['blind_right']} "

            f"-> "

            f"{row['best_action']}\n"
        )

print(
    "\nSaved:"
)

print(
    "outputs/autonomous_policy.txt"
)