import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# -----------------------------
# 1. Generate Synthetic Dataset
# -----------------------------
np.random.seed(42)

data_size = 200

servers = np.random.randint(50, 1000, data_size)
workload = np.random.uniform(0.3, 1.0, data_size)  # normalized
energy = np.random.uniform(100, 1000, data_size)
temperature = np.random.uniform(20, 90, data_size)

# Efficiency formula (simulated)
efficiency = (
    100
    - (servers * 0.02)
    - (energy * 0.01)
    - (temperature * 0.3)
    + (workload * 20)
)

efficiency = np.clip(efficiency, 10, 100)

df = pd.DataFrame({
    "servers": servers,
    "workload": workload,
    "energy": energy,
    "temperature": temperature,
    "efficiency": efficiency
})

# -----------------------------
# 2. Train ML Model
# -----------------------------
X = df[["servers", "workload", "energy", "temperature"]]
y = df["efficiency"]

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# 3. Optimization Logic
# -----------------------------
def optimize(servers, workload, energy, temperature, efficiency):
    suggestion = []

    if efficiency < 70:
        suggestion.append("Reduce number of servers")

    if temperature > 70:
        suggestion.append("Improve cooling / redistribute load")

    if energy > 700:
        suggestion.append("Optimize energy usage")

    if workload < 0.5:
        suggestion.append("Servers are underutilized")

    if not suggestion:
        suggestion.append("System is already optimized")

    # Example adjustment
    new_servers = int(servers * 0.85) if efficiency < 70 else servers
    new_temp = temperature - 10 if temperature > 70 else temperature

    return suggestion, new_servers, new_temp

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("⚡ AdaptX: AI Resource Optimization System")

st.sidebar.header("Input Parameters")

servers_input = st.sidebar.slider("Number of Servers", 50, 2000, 500)
workload_input = st.sidebar.slider("Workload (0–1)", 0.0, 1.0, 0.7)
energy_input = st.sidebar.slider("Energy Consumption", 100, 1500, 600)
temperature_input = st.sidebar.slider("Temperature (°C)", 20, 100, 60)

# -----------------------------
# 5. Prediction
# -----------------------------
input_data = np.array([[servers_input, workload_input, energy_input, temperature_input]])
predicted_efficiency = model.predict(input_data)[0]

# -----------------------------
# 6. Optimization
# -----------------------------
suggestions, new_servers, new_temp = optimize(
    servers_input,
    workload_input,
    energy_input,
    temperature_input,
    predicted_efficiency
)

# -----------------------------
# 7. Output
# -----------------------------
st.subheader("Predicted Efficiency")
st.write(f"{predicted_efficiency:.2f}%")

st.subheader("Suggestions")
for s in suggestions:
    st.write(f"- {s}")

st.subheader("Optimized Configuration")
st.write(f"Recommended Servers: {new_servers}")
st.write(f"Estimated Temperature: {new_temp:.2f} °C")

# -----------------------------
# 8. Visualization (Optional)
# -----------------------------
st.subheader("📈 Sample Data Preview")
st.dataframe(df.head())