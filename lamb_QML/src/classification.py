from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 12345


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# In[21]:


df = pd.read_csv("/home/fcsc/smerino/quantum/QSVC/Santi/corderos.csv")

df.head()

# In[22]:


w = df["weight"].astype(float).to_numpy()

q1, q2 = np.nanquantile(w, [1/3, 2/3])

y = np.where(w <= q1, 0, np.where(w <= q2, 1, 2)).astype(int)

class_names = ["low", "mid", "high"]


# In[23]:


plt.figure(figsize=(7,4))
plt.hist(w, bins=30)
plt.axvline(q1, color="r")
plt.axvline(q2, color="r")
plt.title("Distribución del peso")
plt.tight_layout()
plt.savefig("pesos.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()


# In[24]:


drop_cols = {"filename", "id", "weight", "Unnamed: 0", "good", "augmented", "train_set", "train"}

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]

#print(feature_cols)

#feature_cols = [
#  "height",
#  "area",
#  "mayor_axis",
#  "minor_axis",
#  "perimeter",
#  "eccentricity",
#  "symmetry",
#]

#print(feature_cols)
# In[25]:


X = df[feature_cols].select_dtypes(include=[np.number]).to_numpy(dtype=float)


# In[26]:


mask = np.isfinite(X).all(axis=1) & np.isfinite(y)

df = df.loc[mask].reset_index(drop=True)
X = X[mask]
y = y[mask]


# In[27]:


from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, test_size=0.2,
)

# In[28]:

# Guardamos copia para el baseline clásico antes de escalar a (0, pi)
train_features_raw = train_features.copy()
test_features_raw = test_features.copy()

scaler = MinMaxScaler(feature_range=(0, np.pi))
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

num_classes = len(class_names) 
tr_counts = np.zeros(num_classes, dtype=int) 
te_counts = np.zeros(num_classes, dtype=int) 
c, n = np.unique(train_labels, return_counts=True); tr_counts[c] = n 
c, n = np.unique(test_labels, return_counts=True); te_counts[c] = n

x = np.arange(num_classes) 
plt.figure(figsize=(7, 4)) 
plt.bar(x, tr_counts, label="train") 
plt.bar(x, te_counts, bottom=tr_counts, label="test") 
plt.xticks(x, class_names) 
plt.title("M4 — Train/Test class distribution") 
plt.xlabel("class") 
plt.ylabel("count") 
plt.legend() 
plt.tight_layout() 
plt.savefig("Train_Test_distribution.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show() 

pca = PCA(n_components=2, random_state=algorithm_globals.random_seed) 
train_2d = pca.fit_transform(train_features) 
test_2d = pca.transform(test_features)

plt.figure(figsize=(6, 6)) 
for c in range(num_classes): 
    plt.scatter(train_2d[train_labels == c, 0], train_2d[train_labels == c, 1], marker="s", facecolors="none", label=f"{class_names[c]} train") 
    plt.scatter(test_2d[test_labels == c, 0], test_2d[test_labels == c, 1], marker="o", label=f"{class_names[c]} test") 
plt.title("M4 — Dataset (PCA 2D projection)") 
plt.xlabel("PC1") 
plt.ylabel("PC2") 
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left") 
plt.tight_layout() 
plt.savefig("DatasetPCA2D.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()


"""## Baseline clásico: SVC con kernel RBF"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Escalado estándar para el modelo clásico (mejor que usar (0, pi))
scaler_rbf = StandardScaler()
train_rbf = scaler_rbf.fit_transform(train_features_raw)
test_rbf = scaler_rbf.transform(test_features_raw)

svc_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
svc_rbf.fit(train_rbf, train_labels)
pred_rbf = svc_rbf.predict(test_rbf)
score_rbf = accuracy_score(test_labels, pred_rbf)

# Gráfico de accuracy individual
plt.figure(figsize=(5, 4))
plt.bar(["M4 SVC RBF (clásico)"], [score_rbf], color="forestgreen")
plt.ylim(0, 1)
plt.title("M4 — SVC RBF accuracy (baseline clásico)")
plt.ylabel("accuracy")
plt.tight_layout()
plt.savefig("M4AccuracyRBF.png", dpi=300, bbox_inches="tight")
plt.close()

# Matriz de confusión
cm_rbf = confusion_matrix(test_labels, pred_rbf)
disp_rbf = ConfusionMatrixDisplay(cm_rbf, display_labels=class_names)
disp_rbf.plot(cmap=plt.cm.Greens)
plt.title("Confusion matrix (SVC RBF)")
plt.tight_layout()
plt.savefig("Confusion_matrix_RBF.png", dpi=300, bbox_inches="tight")
plt.close()

print("Baseline RBF accuracy:", score_rbf)



"""## Defining the quantum kernel"""
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

m4_dimension = train_features.shape[1]
m4_feature_map = zz_feature_map(feature_dimension=m4_dimension, reps=2, entanglement="linear")

sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
m4_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=m4_feature_map)

"""## Classification with SVC"""

from sklearn.svm import SVC

m4_svc_callable = SVC(kernel=m4_kernel.evaluate)
m4_svc_callable.fit(train_features, train_labels)

pred_callable = m4_svc_callable.predict(test_features)
score_callable = accuracy_score(test_labels, pred_callable)

plt.figure(figsize=(5, 4))
plt.bar(["M4 SVC callable"], [score_callable])
plt.ylim(0, 1)
plt.title("M4 — Callable kernel accuracy")
plt.ylabel("accuracy")
plt.tight_layout()
plt.savefig("M4Accuracy.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()

cm = confusion_matrix(test_labels, pred_callable)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion matrix (SVC)")
plt.tight_layout()
plt.savefig("Confusion_matrix_SVC.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()
"""### Precomputed kernel matrix"""

K_train = m4_kernel.evaluate(x_vec=train_features)
K_test = m4_kernel.evaluate(x_vec=test_features, y_vec=train_features)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(np.asmatrix(K_train), cmap="Blues")
axs[0].set_title("M4 — Training kernel matrix")
axs[1].imshow(np.asmatrix(K_test), cmap="Reds")
axs[1].set_title("M4 — Testing kernel matrix")
plt.tight_layout()
plt.savefig("M4TrainingTesting.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()



m4_svc_pre = SVC(kernel="precomputed")
m4_svc_pre.fit(K_train, train_labels)

pred_pre = m4_svc_pre.predict(K_test)
score_pre = accuracy_score(test_labels, pred_pre)

plt.figure(figsize=(5, 4))
plt.bar(["M4 SVC precomputed"], [score_pre])
plt.ylim(0, 1)
plt.title("M4 — Precomputed kernel accuracy")
plt.ylabel("accuracy")
plt.tight_layout()
plt.savefig("M4SVCprecomputed.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()

cm_pre = confusion_matrix(test_labels, pred_pre)
disp2 = ConfusionMatrixDisplay(cm_pre, display_labels=class_names)
disp2.plot(cmap=plt.cm.Blues)
plt.title("Confusion matrix (SVC)")
plt.tight_layout()
plt.savefig("ConfusionMatrix.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()

"""## Classification with QSVC"""

from qiskit_machine_learning.algorithms import QSVC

m4_qsvc = QSVC(quantum_kernel=m4_kernel)
m4_qsvc.fit(train_features, train_labels)

pred_qsvc = m4_qsvc.predict(test_features)
score_qsvc = accuracy_score(test_labels, pred_qsvc)

plt.figure(figsize=(5, 4))
plt.bar(["M4 QSVC"], [score_qsvc])
plt.ylim(0, 1)
plt.title("M4 — QSVC accuracy")
plt.ylabel("accuracy")
plt.tight_layout()
plt.savefig("M4QSVCAccuracy.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()

cm_qsvc = confusion_matrix(test_labels, pred_qsvc)
disp = ConfusionMatrixDisplay(cm_qsvc, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion matrix (QSVC)")
plt.tight_layout()
plt.savefig("M4QSVCConfusionMatrix.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()

"""## Evaluation of models used for classification"""
plt.figure(figsize=(7, 4))
#plt.bar(["M4 SVC callable", "M4 SVC precomputed", "M4 QSVC"],
#                [score_callable, score_pre, score_qsvc])
plt.bar(["M4 SVC callable", "M4 SVC precomputed", "M4 QSVC", "M4 SVC RBF (clásico)"],
     [score_callable, score_pre, score_qsvc, score_rbf],
     color=["tab:blue", "tab:orange", "tab:purple", "tab:green"])
plt.ylim(0, 1)
plt.title("M4 — Model comparison (accuracy)")
plt.ylabel("accuracy")
plt.tight_layout()
plt.savefig("M4ComparisonAccuracy.png", dpi=300, bbox_inches="tight")
plt.close()
#plt.show()

#print(score_callable, score_pre, score_qsvc)
print(score_callable, score_pre, score_qsvc, score_rbf)
