# 🫀 Predictor de Problemas Cardíacos

Este proyecto es una aplicación en **Streamlit** que permite predecir si una persona sufrirá o no de problemas cardíacos,
basándose en su **edad** y **nivel de colesterol**.  
El modelo fue entrenado con un **SVC** y los datos fueron normalizados con **MinMaxScaler**.

## 🚀 Características
- Entrada de datos mediante sliders:
  - Edad: entre 25 y 80 años (por defecto 55, incremento de 1).
  - Colesterol: entre 120 y 600 (por defecto 242, incremento de 2).
- Predicción binaria:
  - **0 → No sufrirá del corazón** (muestra una imagen positiva).
  - **1 → Sufrirá del corazón** (muestra una imagen de advertencia).
- Título: *Predictor de problemas cardiacos*.
- Subtítulo: *Elaborado por Unab 2025*.

## 📦 Requisitos
Instalar dependencias con:
```bash
pip install -r requirements.txt
