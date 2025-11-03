# 🤖 Proyecto de Predicción de Demanda con Machine Learning

## 👥 Integrantes
- Joel Cabrera
- Carlos Moyaa
- Andres Sanchez
- Maria Maldonado

## 📋 Descripción
Este proyecto implementa un sistema de predicción de tendencias de demanda utilizando técnicas de Machine Learning. Se analizan diferentes factores que influyen en la demanda como precios, promociones, factores estacionales y segmentos de clientes para predecir si la demanda será estable, creciente o decreciente.

## 🎯 Objetivos
* Analizar patrones en datos históricos de ventas
* Implementar modelos de Machine Learning para predicción
* Comparar el rendimiento de diferentes algoritmos
* Proporcionar insights sobre factores que influyen en la demanda

## 🗂️ Estructura del Proyecto
```
src/
├── data/
│   └── demand_forecasting.csv    # Dataset principal
├── docs/
│   └──                 # Documentación del proyecto
└── notebooks/
    ├── EDA.ipynb               # Análisis Exploratorio de Datos
    └── Algoritmos-ML.ipynb     # Implementación de modelos ML
```

## 📊 Modelos Implementados
* 🌲 **Árbol de Decisión**: Modelo base interpretable
* 🎯 **SVM (Support Vector Machine)**: Con optimización de hiperparámetros
* 🌳 **Random Forest**: Modelo ensemble robusto

## 📈 Características Analizadas
* Cantidad de Ventas
* Precios
* Promociones
* Factores Estacionales
* Factores Externos
* Segmentos de Clientes
* Patrones Temporales (Mes, Día de la semana)

## 🔍 Métricas de Evaluación
* Precisión (Precision)
* Exhaustividad (Recall)
* Puntuación F1 (F1-Score)
* Matrices de Confusión

## 🛠️ Tecnologías Utilizadas
* Python 3.x
* Pandas para manipulación de datos
* Scikit-learn para modelos ML
* Seaborn/Matplotlib para visualizaciones

## 📌 Resultados Principales
* Pipeline completo de preprocesamiento de datos
* Comparación experimental de modelos
* Visualizaciones detalladas de resultados
* Análisis de factores importantes en la predicción

## 🚀 Cómo Usar
1. Clone el repositorio
2. Instale las dependencias necesarias
3. Los notebooks están en la carpeta `src/notebooks/`:
   * Revisar `EDA.ipynb` para entender los datos
   * Revisar `Algoritmos-ML.ipynb` para ver la implementación

## 📝 Notas
* Los modelos están optimizados para un equilibrio entre precisión e interpretabilidad
* Se incluye validación cruzada para resultados más robustos
* La comparación de modelos considera múltiples métricas

## 📄 Licencia
Este proyecto está bajo la Licencia MIT - vea el archivo `LICENSE` para más detalles.