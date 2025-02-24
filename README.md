# 🌍 Machine Learning - GDP Prediction of Countries

Este proyecto utiliza **Machine Learning** para predecir el **Producto Interno Bruto (PIB)** de diferentes países en función de diversas características socioeconómicas. Se emplean modelos de **Regresión Lineal, Random Forest y Gradient Boosting** para analizar la relación entre variables y mejorar la precisión de las predicciones.

## 📊 Modelos Utilizados
El análisis se basa en tres modelos de Machine Learning:

### 1️⃣ **Regresión Lineal (Linear Regression)**
   - Modelo simple que establece una relación lineal entre las variables independientes y el PIB.
   - Se utiliza como referencia inicial para evaluar la capacidad predictiva de las características seleccionadas.
   - Implementado con `sklearn.linear_model.LinearRegression`.

### 2️⃣ **Random Forest**
   - Un modelo basado en múltiples árboles de decisión que combinan predicciones para mejorar la precisión.
   - Es robusto ante valores atípicos y captura relaciones no lineales entre las variables.
   - Implementado con `sklearn.ensemble.RandomForestClassifier`.

### 3️⃣ **Gradient Boosting**
   - Un modelo avanzado que entrena secuencialmente árboles de decisión, corrigiendo errores en cada iteración.
   - Mejora la precisión, pero es más costoso computacionalmente.
   - Implementado con `sklearn.ensemble.GradientBoostingClassifier`.

## 🛠️ Instalación y Dependencias
Para ejecutar el proyecto en tu máquina, instala las siguientes dependencias:

```sh
pip install pandas scikit-learn matplotlib seaborn jupyter
```# ml_all_countries
