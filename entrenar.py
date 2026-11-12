"""Entrena el predictor de problemas cardíacos (SVC) y guarda scaler.jb y svc_model.jb.

Uso:
    python entrenar.py                 # usa pacientes.csv de esta carpeta
    python entrenar.py ruta/datos.csv

Pasos:
  1. Carga pacientes.csv (edad, colesterol, problema_cardiaco).
  2. Elimina las filas con valores faltantes (4 de 302).
  3. Separa entrenamiento/prueba 80/20 (random_state=42, estratificado).
  4. Normaliza edad y colesterol a [0, 1] con MinMaxScaler.
  5. Entrena un SVC con kernel polinómico (C=0.8) y reporta métricas en prueba.
  6. Guarda el escalador y el modelo con joblib (los archivos que carga app.py).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

AQUI = Path(__file__).parent
csv = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI / "pacientes.csv"

df = pd.read_csv(csv, encoding="utf-8-sig").dropna()
X = df[["edad", "colesterol"]].values
y = df["problema_cardiaco"].astype(int).values
print(f"Registros usados: {len(df)}  (clase 1: {y.sum()}, clase 0: {(y == 0).sum()})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = MinMaxScaler().fit(X_train)
modelo = SVC(C=0.8, kernel="poly", probability=True, random_state=42)
modelo.fit(scaler.transform(X_train), y_train)

pred = modelo.predict(scaler.transform(X_test))
print(f"\nExactitud en prueba: {accuracy_score(y_test, pred):.3f}")
print("Matriz de confusión (filas = real, columnas = predicho):")
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["Sin problema (0)", "Con problema (1)"]))

joblib.dump(scaler, AQUI / "scaler.jb")
joblib.dump(modelo, AQUI / "svc_model.jb")
print("Guardados: scaler.jb y svc_model.jb")
