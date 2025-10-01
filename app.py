import streamlit as st
import joblib
import numpy as np

# Cargar scaler y modelo
scaler = joblib.load("scaler.jb")
model = joblib.load("svc_model.jb")

# Título y subtítulo
st.title("🫀 Predictor de Problemas Cardiacos")
st.subheader("Elaborado por Unab 2025")

# Sliders para los datos de entrada
edad = st.slider("Edad", min_value=25, max_value=80, value=55, step=1)
colesterol = st.slider("Colesterol", min_value=120, max_value=600, value=242, step=2)

# Botón de predicción
if st.button("Predecir"):
    # Preparar datos
    X = np.array([[edad, colesterol]])
    X_scaled = scaler.transform(X)

    # Predicción
    pred = model.predict(X_scaled)[0]

    if pred == 0:
        st.success("✅ No sufrirá del corazón")
        st.image("https://img.freepik.com/foto-gratis/feliz-mujer-atractiva-bailando-divirtiendose-levantando-manos-preocupaciones-disfrutando-musica-pie-contra-pared-blanca_176420-38816.jpg",
                 caption="Salud en buen estado", use_column_width=True)
    else:
        st.error("⚠️ Sufrirá del corazón")
        st.image("https://th.bing.com/th/id/OIP.gvXGXrJ3RKcn0khzYqFc7QHaD4?w=309&h=180&c=7&r=0&o=7&pid=1.7&rm=3",
                 caption="Posible riesgo cardíaco", use_column_width=True)
