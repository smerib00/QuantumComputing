from qiskit_machine_learning.utils import algorithm_globals
algorithm_globals.random_seed = 12345

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

# Qiskit
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# 1. CARGA Y PREPARACIÓN
df = pd.read_csv("/home/fcsc/smerino/quantum/QSVC/Santi/corderos.csv")

# Variable objetivo continua: peso real
y_reg = df["weight"].astype(float).to_numpy()

# Características (todo lo numérico excepto metadatos)
drop_cols = {"filename", "id", "weight", "Unnamed: 0", "good", "augmented", "train_set", "train"}
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].select_dtypes(include=[np.number]).to_numpy(dtype=float)

# Limpiar NaNs / Inf en X o en el peso
mask = np.isfinite(X).all(axis=1) & np.isfinite(y_reg)
X = X[mask]
y_reg = y_reg[mask]

# Split único
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=algorithm_globals.random_seed
)

# 2. ESCALADO
# Clásico: StandardScaler (mejor para SVR RBF)
scaler_classic = StandardScaler()
X_train_classic = scaler_classic.fit_transform(X_train)
X_test_classic = scaler_classic.transform(X_test)

# Cuántico: MinMax a (0, π) (el feature map espera ese rango)
scaler_quantum = MinMaxScaler(feature_range=(0, np.pi))
X_train_q = scaler_quantum.fit_transform(X_train)
X_test_q = scaler_quantum.transform(X_test)

# 3. KERNEL CUÁNTICO
feature_map = zz_feature_map(
    feature_dimension=X_train_q.shape[1],
    reps=2,
    entanglement="linear"
)
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# 4. BASELINE CLÁSICO (SVR RBF)
svr = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)
svr.fit(X_train_classic, y_train)
pred_reg = svr.predict(X_test_classic)

mae = mean_absolute_error(y_test, pred_reg)
rmse = np.sqrt(mean_squared_error(y_test, pred_reg))
r2 = r2_score(y_test, pred_reg)

print("SVR clásico")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# 5. REGRESIÓN CUÁNTICA (QSVR)
qsvr = SVR(kernel=quantum_kernel.evaluate, C=10, epsilon=0.1)
qsvr.fit(X_train_q, y_train)
pred_q = qsvr.predict(X_test_q)

mae_q = mean_absolute_error(y_test, pred_q)
rmse_q = np.sqrt(mean_squared_error(y_test, pred_q))
r2_q = r2_score(y_test, pred_q)

print("\nSVR cuántico")
print("MAE :", mae_q)
print("RMSE:", rmse_q)
print("R²  :", r2_q)

# 6. COMPARATIVA R²
plt.figure(figsize=(6, 4))
plt.bar(["SVR clásico", "SVR cuántico"], [r2, r2_q], color=["forestgreen", "tab:purple"])
plt.ylabel("R²")
plt.title("Comparación de regresión")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("Regression_R2.png", dpi=300, bbox_inches="tight")
plt.close()

# 7. REAL vs PREDICHO (clásico)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_reg, alpha=0.7, edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Peso real")
plt.ylabel("Peso predicho")
plt.title("SVR clásico — Real vs Predicho")
plt.tight_layout()
plt.savefig("SVR_prediction.png", dpi=300, bbox_inches="tight")
plt.close()

# 8. REAL vs PREDICHO (cuántico)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_q, alpha=0.7, edgecolors="k", color="tab:purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Peso real")
plt.ylabel("Peso predicho")
plt.title("SVR con kernel cuántico — Real vs Predicho")
plt.tight_layout()
plt.savefig("QuantumSVR_prediction.png", dpi=300, bbox_inches="tight")
plt.close()

# 9. RESUMEN
print("\n==========================")
print("RESULTADOS REGRESIÓN")
print("==========================")
print(f"SVR clásico  : MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
print(f"SVR cuántico : MAE={mae_q:.3f} RMSE={rmse_q:.3f} R²={r2_q:.3f}")
