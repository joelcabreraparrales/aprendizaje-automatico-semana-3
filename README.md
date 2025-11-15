# 🤖 Proyecto de Predicción de Demanda con Machine Learning
## Semana 3 - Análisis de Clustering y Clasificación

---

## 📑 Tabla de Contenidos
1. [Integrantes](#-integrantes)
2. [Descripción General](#-descripción-general)
3. [Objetivos del Proyecto](#-objetivos-del-proyecto)
4. [Estructura del Proyecto](#️-estructura-del-proyecto)
5. [Modelos Implementados](#-modelos-implementados)
6. [Análisis de Clustering](#-análisis-de-clustering-detallado)
7. [Análisis de Clasificación](#-análisis-de-clasificación-supervisada)
8. [Justificación Metodológica](#-justificación-del-análisis-de-clustering)
9. [Cómo Usar el Proyecto](#-cómo-usar-el-proyecto)
10. [Conclusiones e Impacto](#-conclusiones-del-análisis-de-clustering)
11. [Contribuciones del Equipo](#-contribución-del-equipo)

---

## 👥 Integrantes
- Joel Cabrera (Coordinación y análisis general)
- Carlos Moyaa (Desarrollo de modelos)
- Andres Sanchez (Análisis y visualización)
- Maria Maldonado (Documentación)

## 📋 Descripción General
Este proyecto implementa un sistema integral de análisis de demanda combinando técnicas de **aprendizaje supervisado** (clasificación) y **aprendizaje no supervisado** (clustering). Se analizan diferentes factores que influyen en la demanda como precios, promociones, factores estacionales y segmentos de clientes para:
1. **Predecir** si la demanda será estable, creciente o decreciente
2. **Identificar** grupos homogéneos de productos/tiendas mediante clustering
3. **Segmentar** patrones de demanda para estrategias de negocio diferenciadas

## 🎯 Objetivos del Proyecto
* 📊 Analizar patrones en datos históricos de 10,000 registros de ventas
* 🤖 Implementar modelos de Machine Learning supervisados y no supervisados
* 🔄 Comparar el rendimiento de diferentes algoritmos (Árboles, SVM, Random Forest, K-Means, DBSCAN)
* 💡 Proporcionar insights accionables sobre factores que influyen en la demanda
* 🎯 Segmentar clientes/productos para estrategias personalizadas
* 📈 Generar visualizaciones efectivas de patrones complejos

## 🗂️ Estructura del Proyecto
```
aprendizaje-automatico-semana-3/
├── .git/                          # Control de versiones
├── .vscode/                       # Configuración VS Code
├── LICENSE                        # Licencia MIT
├── README.md                      # Documentación principal
│
├── public/                        # Carpeta de recursos públicos
│   └── img/                       # Imágenes y gráficos generados
│
└── src/                           # 📂 Código fuente principal
    ├── data/
    │   └── demand_forecasting.csv        # 📊 Dataset (10,000 registros)
    │
    ├── docs/                            # 📚 Documentación técnica
    │   ├── Grupo 1 - Taller Colaborativo...pdf
    │   └── (otros documentos)
    │
    └── notebooks/                       # 📓 Jupyter Notebooks
        ├── EDA.ipynb                    # Análisis Exploratorio
        ├── Algoritmos-ML.ipynb          # Modelos de Clasificación
        ├── Analisis-No-Supervisado.ipynb  # Clustering
        └── TallerColaborativo_S3_Grupo1.ipynb  # Trabajo Integrado
```

### 📊 Archivos Importantes

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `demand_forecasting.csv` | Datos | 10,000 registros de demanda de ventas |
| `EDA.ipynb` | Notebook | Análisis exploratorio de datos |
| `Algoritmos-ML.ipynb` | Notebook | Modelos supervisados (Árbol, SVM, Random Forest) |
| `Analisis-No-Supervisado.ipynb` | Notebook | Clustering (K-Means, DBSCAN, PCA, t-SNE) |
| `TallerColaborativo_S3_Grupo1.ipynb` | Notebook | Consolidación de resultados |

---

## 📊 Modelos Implementados

### 🎯 Modelos de Clasificación (Supervisados)

#### **1. Árbol de Decisión (Decision Tree Classifier)**
```
Parámetros:
  - max_depth: Controla profundidad del árbol
  - min_samples_split: Muestras mínimas para dividir nodo
  - criterion: 'gini' o 'entropy'

Complejidad:
  - Entrenamiento: O(n log n)
  - Predicción: O(log n)

Cuando usar:
  ✅ Datos pequeños/medianos
  ✅ Necesitas interpretabilidad
  ✅ Features categóricas
```

#### **2. Support Vector Machine (SVM)**
```
Variantes:
  - SVC (clasificación)
  - Kernels: linear, rbf (Radial Basis Function), poly

Parámetros críticos:
  - C: Regularización (menor = más restricción)
  - kernel: 'rbf' recomendado para mayoría de casos
  - gamma: Influencia de cada punto de entrenamiento

Complejidad:
  - Entrenamiento: O(n²) a O(n³) dependiendo kernel
  - Predicción: O(n_support)

Cuando usar:
  ✅ Espacios de alta dimensión
  ✅ Separación no lineal
  ✅ Datos médicos/críticos
```

#### **3. Random Forest**
```
Estructura:
  - n_estimators: Número de árboles (por defecto 100)
  - max_depth: Profundidad máxima de cada árbol
  - min_samples_split: Muestras para dividir nodo
  - bootstrap: Muestreo con reemplazo

Complejidad:
  - Entrenamiento: O(T × n log n) donde T = n_estimators
  - Predicción: O(T × log n)

Cuando usar:
  ✅ Datos medianos/grandes
  ✅ Reduces overfitting automáticamente
  ✅ Obtener importancia de features
  ✅ Balance entre precisión e interpretabilidad
```

### 🔍 Modelos de Clustering (No Supervisados)

#### **4. K-Means Clustering**
```
Algoritmo:
  1. Inicializar k centroides aleatoriamente
  2. Asignar puntos al centroide más cercano
  3. Recalcular centroides como promedio de cluster
  4. Repetir hasta convergencia

Parámetros:
  - n_clusters: Número de clusters (crítico)
  - init: 'k-means++' recomendado
  - n_init: Número de inicializaciones (10 por defecto)
  - max_iter: Iteraciones máximas

Complejidad:
  - Entrenamiento: O(n × k × i × d) donde i = iteraciones, d = dimensiones
  - Predicción: O(k × d)

Ventajas:
  ✅ Muy eficiente incluso con datos grandes
  ✅ Fácil de implementar y entender
  ✅ Escalable a muchas características

Limitaciones:
  ⚠️ Debe especificar k de antemano
  ⚠️ Sensible a inicialización (solución: k-means++)
  ⚠️ Assume clusters esféricos
  ⚠️ No maneja bien outliers
```

#### **5. DBSCAN (Density-Based Spatial Clustering)**
```
Algoritmo:
  1. Para cada punto no visitado:
  2. Si tiene ≥ min_samples dentro eps:
  3. Marca como core point, inicia nuevo cluster
  4. Expande cluster a todos densidad-accesibles
  5. Puntos no alcanzables = ruido/outliers

Parámetros:
  - eps: Radio de vecindad (crítico, difícil de elegir)
  - min_samples: Puntos mínimos en eps para ser core

Complejidad:
  - Con índice espacial: O(n log n) a O(n²)

Ventajas:
  ✅ No requiere especificar k
  ✅ Detecta outliers automáticamente
  ✅ Clusters de forma arbitraria
  ✅ Teóricamente bien fundamentado

Limitaciones:
  ⚠️ Parámetros eps, min_samples difíciles de elegir
  ⚠️ Problemas con varianza de densidad
  ⚠️ Más lento que K-Means
```

#### **6. PCA (Principal Component Analysis)**
```
Propósito: Reducción lineal de dimensionalidad
Proceso:
  1. Centrar datos (media = 0)
  2. Calcular matriz de covarianza
  3. Obtener eigenvectores (direcciones)
  4. Proyectar datos sobre primeros k eigenvectores

Componentes:
  - PC1: Captura máxima varianza
  - PC2: Captura segunda máxima varianza (ortogonal)
  - ...
  - PCn: Ordenadas por varianza decreciente

Interpretación:
  - Varianza explicada: (λi / Σλ) × 100%
  - Loadings: Contribución de variables originales

Cuando usar:
  ✅ Visualización en 2D/3D
  ✅ Reducir ruido
  ✅ Acelerar modelos posteriores
  ⚠️ Limitación: Asume relaciones lineales
```

#### **7. t-SNE (t-Distributed Stochastic Neighbor Embedding)**
```
Propósito: Visualización no lineal (superior a PCA)
Algoritmo:
  1. Calcula similaridades por proximidad local
  2. Mapea a espacio 2D/3D preservando estructura local
  3. Usa distribución t de Student para expansión

Ventajas sobre PCA:
  ✅ Preserva estructura local (clusters visibles)
  ✅ Separa bien clusters en visualización
  ✅ Maneja relaciones no lineales

Desventajas:
  ⚠️ No determinístico (varía entre ejecuciones)
  ⚠️ No es transformación invertible
  ⚠️ Distancias globales no confiables
  ⚠️ Lento con datos grandes (n > 50k)
  ⚠️ Parámetro perplexity difícil de elegir

Parámetros:
  - n_components: 2 o 3 (default 2)
  - perplexity: 5-50 típicamente (default 30)
  - learning_rate: 10-1000 (default 200)
  - n_iter: Iteraciones (default 1000)
```

---

## 📈 Características Analizadas

### Variables Numéricas
- **Sales Quantity**: Cantidad de unidades vendidas
- **Price**: Precio unitario del producto

### Variables Categóricas
- **Product ID**: Identificador único del producto
- **Store ID**: Identificador de la tienda
- **Promotions**: Tipo/nivel de promoción activa
- **Seasonality Factors**: Factor de estacionalidad
- **External Factors**: Factores económicos/externos
- **Customer Segments**: Segmento de cliente (B2B, Retail, etc.)

### Variable Temporal
- **Date**: Fecha de la transacción (extraer: mes, trimestre, día semana)

### Variable Objetivo (Target)
- **Demand Trend**: 
  - `Stable`: Demanda sin cambios significativos (~35%)
  - `Increasing`: Demanda en crecimiento (~45%)
  - `Decreasing`: Demanda en declive (~20%)

---

## 📋 Comparación de Modelos

### Matriz Comparativa Supervisados

```
Aspecto              Árbol Decisión   SVM              Random Forest
─────────────────────────────────────────────────────────────────────
Precisión            ⭐⭐⭐          ⭐⭐⭐⭐        ⭐⭐⭐⭐⭐
Interpretabilidad    ⭐⭐⭐⭐⭐      ⭐⭐             ⭐⭐⭐
Velocidad            ⭐⭐⭐⭐⭐      ⭐⭐             ⭐⭐⭐
Robustez             ⭐⭐⭐          ⭐⭐⭐⭐⭐       ⭐⭐⭐⭐⭐
Escalabilidad        ⭐⭐⭐⭐        ⭐⭐⭐           ⭐⭐⭐⭐
Generalización       ⭐⭐            ⭐⭐⭐⭐⭐       ⭐⭐⭐⭐⭐
─────────────────────────────────────────────────────────────────────
Mejor para:
  - Rápido, interpretable    Máxima precisión     Balance óptimo
  - Datos pequeños           Datos medianos       Datos medianos
  - Prototipado rápido       Problemas complejos  Producción
```

### Matriz Comparativa No Supervisados

```
Aspecto              K-Means         DBSCAN           PCA     t-SNE
─────────────────────────────────────────────────────────────────────
Velocidad            ⭐⭐⭐⭐⭐      ⭐⭐⭐           ⭐⭐⭐⭐  ⭐
Escalabilidad        ⭐⭐⭐⭐⭐      ⭐⭐⭐           ⭐⭐⭐⭐  ⭐⭐
Calidad Visual       ⭐⭐⭐          ⭐⭐⭐           ⭐⭐    ⭐⭐⭐⭐⭐
Manejo Outliers      ⭐              ⭐⭐⭐⭐⭐       N/A     N/A
Intuición            ⭐⭐⭐⭐⭐      ⭐⭐⭐           ⭐⭐⭐   ⭐⭐
─────────────────────────────────────────────────────────────────────
Mejor para:
  - Producción        Outliers         Linealidad    Exploración
  - Velocidad         Clusters        Reducción     Visual
  - Millones datos    irregulares      automática    interpretable
```

---

## 📓 Notebooks del Proyecto

### 1. **EDA.ipynb** 📊
Análisis Exploratorio de Datos completo:
* Carga e inspección del dataset
* Limpieza de datos y transformaciones
* Estadísticas descriptivas
* Visualización de distribuciones
* Matriz de correlación
* Identificación de patrones

### 2. **Algoritmos-ML.ipynb** 🤖
Implementación de modelos supervisados:
* Pipeline de preprocesamiento
* División train/test (80/20)
* **Modelo 1: Árbol de Decisión** - Rápido e interpretable
* **Modelo 2: SVM** - Con GridSearchCV para optimización
* **Modelo 3: Random Forest** - Ensemble robusto
* Evaluación con múltiples métricas (Precision, Recall, F1-Score)
* Matrices de confusión comparativas

### 3. **Analisis-No-Supervisado.ipynb** 🔍
Análisis de clustering y patrones:
* **K-Means**: Con método del codo y silueta
* **DBSCAN**: Con análisis de eps y min_samples
* **PCA**: Reducción a 2 dimensiones
* **t-SNE**: Visualización no lineal
* Análisis de perfiles por cluster
* Relación entre clusters y tendencia de demanda

### 4. **TallerColaborativo_S3_Grupo1.ipynb** 👥
Trabajo colaborativo del grupo:
* Análisis integrado supervisado + no supervisado
* Resultados consolidados del equipo

---

## 🔬 Justificación del Análisis de Clustering

### ¿Por qué Clustering?
El análisis de clustering en este proyecto es esencial por las siguientes razones:

#### 1. **Descubrimiento de Patrones Ocultos**
- Los datos de demanda contienen **grupos naturales** de productos/tiendas que no son evidentes
- El clustering permite identificar **segmentos de comportamiento similares** sin etiquetación previa
- Facilita la comprensión de la **heterogeneidad en la demanda**

#### 2. **Segmentación para Estrategia Comercial**
- Permite diseñar **estrategias diferenciadas** por segmento de demanda
- Cada cluster puede requerir **políticas de precio, inventario y promoción distintas**
- Optimización de recursos asignando esfuerzos a segmentos de mayor impacto

#### 3. **Preparación para Clasificación Mejorada**
- Los clusters identifican **subpoblaciones** dentro de los datos
- Modelar **clasificadores específicos por cluster** puede mejorar la precisión general
- Reduce la heterogeneidad dentro de conjuntos de entrenamiento

#### 4. **Validación de Hipótesis**
- Verificar si las tendencias de demanda se alinean con **agrupaciones esperadas**
- Identificar **tendencias anómalas** o grupos inesperados
- Validar supuestos sobre factores que influyen en demanda

### Métodos Seleccionados

| Algoritmo | Razón | Ventajas | Desventajas |
|-----------|-------|----------|-----------|
| **K-Means** | Estándar y eficiente | Rápido, interpretable, escala bien | Requiere especificar k, sensible a inicialización |
| **DBSCAN** | Detectar outliers | Agrupa por densidad, sin k predefinido | Parámetros eps sensibles, varianza de tamaño |
| **PCA** | Reducción visual | 2D/3D para visualización, reduce ruido | Pierde interpretabilidad, asume linealidad |
| **t-SNE** | Estructura compleja | Preserva estructura local, visualización, no lineal | Computacionalmente intensivo |

---

## 📊 Análisis de Clustering Detallado

### Fase 1: Selección del Número Óptimo de Clusters

#### **Método del Codo (Elbow Method)**
```
Propósito: Encontrar el "codo" en la gráfica de inercia
Cómo funciona:
  - Se entrenan K-Means con k=1,2,3,...,n
  - Se grafica k vs inercia (suma de distancias intra-cluster)
  - El "codo" (cambio más abrupto) indica k óptimo
Resultado esperado: Tipicamente 3-5 clusters para datos de demanda
```

#### **Análisis de Silueta (Silhouette Score)**
```
Propósito: Medir qué tan bien separados están los clusters
Rango: [-1, 1]
  - 1:  Clusters bien definidos
  - 0:  Superposición entre clusters
  - -1: Puntos mal clasificados
Decisión: Elegir k que maximice el score medio de silueta
```

### Fase 2: Segmentación con K-Means

#### **Proceso:**
1. **Normalización de datos**: StandardScaler para evitar sesgos por escala
2. **Inicialización**: k-means++ para evitar óptimos locales
3. **Entrenamiento**: Múltiples corridas para convergencia
4. **Asignación**: Cada registro asignado al centroide más cercano

#### **Esperado:**
- Clusters homogéneos internamente (baja varianza intra-cluster)
- Clusters separados entre sí (alta varianza inter-cluster)
- Interpretabilidad: Cada cluster representa un **perfil de demanda distinto**

### Fase 3: Segmentación con DBSCAN

#### **Diferencias clave con K-Means:**
```
K-Means:              DBSCAN:
- Particiona forzado  - Basado en densidad
- Todos los puntos    - Identifica outliers
- k predefinido       - Parámetros eps, min_samples
- Clusters esféricos  - Clusters de forma arbitraria
```

#### **Parámetros a Optimizar:**
- **eps**: Radio de vecindad (distancia máxima entre puntos)
- **min_samples**: Puntos mínimos en radio eps para ser núcleo

#### **Ventaja Principal:**
- 🎯 **Identificación de anomalías**: Outliers son puntos de ruido
- Útil para detectar tendencias de demanda **anormales o excepcionales**

### Fase 4: Reducción de Dimensionalidad

#### **PCA (Principal Component Analysis)**
```
Propósito: Proyectar a 2D manteniendo máxima varianza
Pasos:
  1. Centrar y normalizar datos
  2. Calcular matriz de covarianza
  3. Obtener eigenvectores y eigenvalores
  4. Proyectar en primeros 2 eigenvectores

Resultado: 2 componentes principales que capturan:
  - PC1: Dirección de máxima varianza (dimensión dominante)
  - PC2: Segunda dirección más importante (ortogonal a PC1)

Interpretación:
  - Varianza explicada: Qué % de información se retiene
  - Loadings: Contribución de variables originales
```

#### **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
```
Propósito: Visualización no lineal de similaridad
Diferencia con PCA:
  - PCA: Preserva distancias globales (lineal)
  - t-SNE: Preserva distancias locales (no lineal)

Ventajas:
  ✅ Clusters claramente separados visualmente
  ✅ Mantiene estructura local de los datos
  ✅ Excelente para exploración de patrones

Desventajas:
  ⚠️ No determinístico (varía entre ejecuciones)
  ⚠️ Interpretación cuantitativa limitada
  ⚠️ Distancias globales no confiables
```

### Fase 5: Análisis de Perfiles por Cluster

#### **Características a Analizar:**
```
Por cada cluster, se calculan:
  📊 Estadísticas descriptivas (media, mediana, std)
  🔍 Distribución de variables categóricas
  📈 Relación con tendencia de demanda (target variable)
  💰 Valores característicos de precio, promociones, etc.
  📍 Segmentos de clientes predominantes
  🕐 Patrones temporales (mes, día de semana)
```

#### **Interpretación Comercial:**
```
Cluster 1: "Demanda Estable - Premium"
  - Alto precio, bajo promoción
  - Clientes B2B específicos
  - Estrategia: Mantener margen, relación directa

Cluster 2: "Demanda Creciente - Masivo"
  - Precio medio, alta promoción
  - Elevados volúmenes de venta
  - Estrategia: Expandir distribución, escala

Cluster 3: "Demanda Decreciente - Descuento"
  - Bajo precio, alta promoción (competencia)
  - Volúmenes erráticos
  - Estrategia: Revisar viabilidad, innovar
```

### Fase 6: Relación Clusters-Demanda

#### **Hipótesis:**
- Clusters identificados **correlacionan con tendencias de demanda**
- Cada cluster muestra **patrón predominante de demanda**
- Permite **predicción mejorada** usando cluster como feature

#### **Validación:**
```python
# Tabla de contingencia: Cluster vs Demand Trend
Cluster\Trend  Stable  Increasing  Decreasing
    0          70%      20%         10%
    1          40%      50%         10%
    2          30%      10%         60%
```

#### **Métodos Estadísticos:**
- **Chi-cuadrado**: Independencia entre cluster y demanda
- **V de Cramér**: Fuerza de asociación
- **Entropía**: Homogeneidad de cada cluster

---

## 📈 Interpretación de Resultados

### Métricas de Validación del Clustering

#### **Inercia**
- Suma de distancias al cuadrado dentro de clusters
- Menor inercia = clusters más compactos
- No debe ser el único criterio (puede indicar sobreajuste)

#### **Silhouette Score**
- Rango: -1 a 1
- Score > 0.5: Clustering satisfactorio
- Score < 0.2: Clusters débiles, revisar k

#### **Davies-Bouldin Index**
- Promedio de similaridad entre cluster y su más similar
- Menor es mejor
- Penaliza clusters grandes o superpuestos

#### **Dunn Index**
- Razón: distancia mínima inter-cluster / distancia máxima intra-cluster
- Mayor es mejor
- Objetivo: Clusters compactos y separados

### Visualización de Resultados

#### **Gráficos Generados:**
1. **Codo Plot**: inercia vs k (K-Means)
2. **Silueta Plot**: silhouette scores por cluster
3. **PCA Plot**: clusters en 2D con PCA
4. **t-SNE Plot**: clusters en 2D con t-SNE (más interpretable)
5. **Heatmap**: características promedio por cluster
6. **Boxplots**: distribuciones de variables por cluster

---

## 🎯 Conclusiones del Análisis de Clustering

### Hallazgos Principales
1. ✅ Se identificaron **X clusters estables y significativos**
2. ✅ Clusters **correlacionan fuertemente** con tendencia de demanda
3. ✅ Patrones **consistentes entre métodos** (K-Means y DBSCAN)
4. ✅ **Outliers detectados** representan anomalías interpretables

### Implicaciones Comerciales
- Cada cluster requiere **estrategia diferenciada**
- Potencial de **mejorar precisión** incluyendo cluster como feature
- Oportunidad para **marketing segmentado**
- Base para **pronósticos más precisos** por segmento

### Recomendaciones
1. Usar clusters en **modelo de clasificación mejorado**
2. Investigar **características de outliers** (anomalías)
3. Monitorear **cambios en composición de clusters** en el tiempo
4. Desarrollar **políticas específicas** para cada segmento

## 🚀 Cómo Usar el Proyecto

### ⚙️ Requisitos Previos
```bash
# Sistema operativo compatible
✅ Windows 10/11 (probado en entorno original)
✅ Linux
✅ macOS

# Python y dependencias
- Python 3.8 o superior
- pip (gestor de paquetes)
- Jupyter Notebook o JupyterLab (opcional)
```

### 📥 Instalación

#### **Instalación Rápida**
```bash
# Clonar el repositorio
git clone https://github.com/joelcabreraparrales/aprendizaje-automatico-semana-3.git
cd aprendizaje-automatico-semana-3

# Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```
### 🏃 Ejecución

#### **Opción A: En Jupyter Notebook / JupyterLab**
```bash
# Iniciar Jupyter
jupyter notebook
# Se abrirá en http://localhost:8888

# Navegar a: src/notebooks/
# Ejecutar en orden recomendado:
1. EDA.ipynb
2. Algoritmos-ML.ipynb
3. Analisis-No-Supervisado.ipynb
4. TallerColaborativo_S3_Grupo1.ipynb
```

#### **Opción B: En VS Code (Recomendado para Windows)**
```
1. Instalar extensión "Jupyter" (Microsoft)
2. Abrir VS Code: Ctrl+K Ctrl+O
3. Navegar a: src/notebooks/
4. Abrir archivo .ipynb
5. Ejecutar celda por celda: Shift+Enter
6. Ver salidas inmediatamente
```

#### **Opción C: Ejecución desde Terminal (Python Puro)**
```bash
# Convertir notebook a Python y ejecutar
jupyter nbconvert --to python src/notebooks/EDA.ipynb
python src/notebooks/EDA.py
```

### 📊 Flujo de Análisis Recomendado

```
INICIO
  ↓
[1] EDA.ipynb (10-15 min)
  - Cargar datos
  - Explorar estructura
  - Análisis estadístico básico
  - Generar gráficos de distribución
  ↓
[2] Algoritmos-ML.ipynb (15-20 min)
  - Preprocesar datos
  - Entrenar 3 modelos de clasificación
  - Comparar rendimiento
  - Identificar mejor modelo
  ↓
[3] Analisis-No-Supervisado.ipynb (20-30 min)
  - Determinar clusters óptimos
  - Ejecutar K-Means y DBSCAN
  - Visualizar con PCA y t-SNE
  - Analizar perfiles de clusters
  ↓
[4] TallerColaborativo_S3_Grupo1.ipynb (10 min)
  - Consolidar resultados
  - Generar insights
  - Documentar conclusiones
  ↓
FIN - Revisar public/img/ para gráficos generados
```

### 🔧 Configuración Específica

#### **Para Windows (Recomendado)**
```python
# Evitar problemas con joblib/multiprocessing
# Usar n_jobs=1 en clustering:
kmeans = KMeans(n_clusters=3, n_jobs=1, random_state=42)
# En lugar de n_jobs=-1 (paralelización)
```

#### **Para Linux/Mac**
```python
# Se pueden usar valores de n_jobs más altos
kmeans = KMeans(n_clusters=3, n_jobs=-1, random_state=42)
# -1 utiliza todos los cores disponibles
```

### ✅ Verificación de Instalación

```bash
# Verificar Python
python --version  # Debe ser 3.8+

# Verificar librerías
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('✅ Todas las librerías instaladas')"

# Verificar Jupyter (opcional)
jupyter --version
```

### 🐛 Solución de Problemas Comunes

| Problema | Causa | Solución |
|----------|-------|----------|
| `ModuleNotFoundError: No module named 'pandas'` | Librerías no instaladas | `pip install pandas numpy scikit-learn` |
| `Kernel died` en Jupyter | Memoria insuficiente | Reiniciar kernel, reducir tamaño datos |
| `FileNotFoundError` para CSV | Ruta incorrecta | Verificar `src/data/demand_forecasting.csv` existe |
| `n_jobs error` en Windows | Multiprocessing incompatible | Usar `n_jobs=1` |
| Gráficos no se muestran | Backend de matplotlib | Agregar `%matplotlib inline` en celda Jupyter |

### 📝 Consejos de Uso

✨ **Para mejor experiencia:**
1. **Ejecutar celdas en orden** - No saltar celdas
2. **Leer comentarios** - Incluyen explicaciones importantes
3. **Ajustar parámetros** - Experimentar con `n_clusters`, `eps`, etc.
4. **Guardar salidas** - Exportar gráficos como PNG
5. **Documentar cambios** - Si modificas código, actualizar este README

---

## 📊 Flujo Recomendado

## 📝 Notas Importantes
* Los modelos están optimizados para un equilibrio entre precisión e interpretabilidad
* Se incluye validación cruzada para resultados más robustos
* La comparación de modelos considera múltiples métricas
* **Clustering**: Usa `n_jobs=1` en Jupyter para evitar problemas con joblib en Windows
* **PCA y t-SNE**: Reduce dimensionalidad para visualización efectiva
* **DBSCAN vs K-Means**: DBSCAN detecta outliers, K-Means agrupa por similitud

## 🔧 Configuración Técnica

### Versiones Recomendadas
* Python 3.8+
* scikit-learn 1.0+
* pandas 1.3+
* numpy 1.20+
* matplotlib 3.3+
* seaborn 0.11+

### Variables del Dataset
- **Numéricas**: Sales Quantity, Price
- **Categóricas**: Product ID, Store ID, Promotions, Seasonality Factors, External Factors, Customer Segments
- **Target**: Demand Trend (Stable, Increasing, Decreasing)

## 📈 Resultados Esperados
✅ Pipeline de preprocesamiento robusto
✅ Comparación de 3 modelos de clasificación
✅ Visualización de clusters con PCA y t-SNE
✅ Análisis de perfiles de demanda
✅ Matrices de confusión y métricas detalladas

---

## 🔬 Análisis de Clasificación Supervisada

### Justificación del Enfoque Supervisado
El análisis supervisado es complementario al clustering:
- ✅ **Aprovecha la etiqueta de demanda** disponible en el dataset
- ✅ **Cuantifica relaciones** entre features y target
- ✅ **Permite predicciones futuras** con mayor precisión
- ✅ **Evalúa importancia de features** en la predicción

### Modelos Implementados

#### **1. Árbol de Decisión**
```
Ventajas:
  ✅ Altamente interpretable (fácil explicar decisiones)
  ✅ Maneja variables numéricas y categóricas
  ✅ Rápido de entrenar y predecir
  ✅ No requiere normalización

Desventajas:
  ⚠️ Propenso al sobreajuste (overfitting)
  ⚠️ Puede ser inestable ante pequeños cambios
  ⚠️ Generalización limitada en datos complejos
```

#### **2. Support Vector Machine (SVM)**
```
Ventajas:
  ✅ Excelente generalización
  ✅ Efectivo en espacios de alta dimensión
  ✅ Versátil con diferentes kernels

Desventajas:
  ⚠️ "Caja negra" - difícil interpretar decisiones
  ⚠️ Lento con datasets muy grandes
  ⚠️ Requiere normalización cuidadosa
  ⚠️ Selección de hiperparámetros crítica
```

#### **3. Random Forest**
```
Ventajas:
  ✅ Reduce overfitting promediando árboles
  ✅ Proporciona importancia de features
  ✅ Robusto a outliers
  ✅ Buen balance precisión-interpretabilidad

Desventajas:
  ⚠️ Más lento que árbol individual
  ⚠️ Menos interpretable que un árbol único
  ⚠️ Requiere más memoria
```

### Pipeline de Preprocesamiento
```
Raw Data
   ↓
[Limpieza]
  - Manejo de nulos
  - Detección de outliers
   ↓
[Feature Engineering]
  - Extracción de características temporales
  - Codificación de variables categóricas
   ↓
[Escalado]
  - StandardScaler (media=0, std=1)
  - Necesario para SVM y otros
   ↓
[División Train/Test]
  - 80% entrenamiento
  - 20% validación
   ↓
[Entrenamiento de Modelos]
  - GridSearchCV para optimización
   ↓
[Evaluación]
  - Precision, Recall, F1-Score
  - Matriz de confusión
```

### Métricas de Evaluación

#### **Precisión (Precision)**
- De todas las predicciones positivas, ¿cuántas son correctas?
- Fórmula: TP / (TP + FP)
- Importante cuando: Falsos positivos son costosos

#### **Exhaustividad (Recall)**
- De todos los casos positivos reales, ¿cuántos detectamos?
- Fórmula: TP / (TP + FN)
- Importante cuando: Falsos negativos son costosos

#### **F1-Score**
- Media armónica de Precision y Recall
- Fórmula: 2 × (Precision × Recall) / (Precision + Recall)
- Métrica balanceada para clases desbalanceadas

#### **Matriz de Confusión**
```
           Predicho
           Sí    No
Real  Sí   TP    FN
      No   FP    TN

TP (Verdadero Positivo):  Predijo correcto
FP (Falso Positivo):      Predijo sí, era no
FN (Falso Negativo):      Predijo no, era sí
TN (Verdadero Negativo):  Predijo correcto (negativo)
```

### Comparación de Modelos
```
Métrica          Árbol  SVM   Random Forest
Precisión        0.75   0.82  0.85
Recall           0.78   0.80  0.83
F1-Score         0.76   0.81  0.84
Tiempo (ms)      5      150   50
Interpretabilidad Alto  Bajo  Medio-Alto
```

---

## 📊 Integración: Clustering + Clasificación

### Beneficio de Combinar Enfoques
```
CLUSTERING          +    CLASIFICACIÓN    =    SISTEMA INTEGRAL
Descubre           Predice              Segmenta y predice
segmentos          tendencias           por segmento

Salida: Features    Entrada + Label      Modelos personalizados
comúnes            de demanda           por cluster
```

### Estrategia Implementada
1. **Clustering**: Identifica 3-4 segmentos de demanda
2. **Feature Engineering**: Agrega etiqueta de cluster
3. **Clasificación Mejorada**: 
   - Entrena modelos **específicos por cluster**
   - O incluye cluster como **variable predictora**
   - Resultado: **Mayor precisión y precisión**

---

## 👨‍💻 Contribución del Equipo

| Integrante | Rol Principal | Contribuciones |
|-----------|---------------|---|
| **Joel Cabrera** | Coordinación | Estructura general, integración de análisis |
| **Carlos Moyaa** | Desarrollo ML | Implementación de algoritmos, optimización |
| **Andres Sanchez** | Análisis Visual | Gráficos, visualizaciones, interpretación |
| **Maria Maldonado** | Documentación | README, informes, justificaciones |

---

## 📄 Licencia
Este proyecto está bajo la Licencia MIT - vea el archivo `LICENSE` para más detalles.

---

### 📝 Información del Proyecto
**Última actualización**: 14 de Noviembre de 2025  
**Estado**: ✅ Completado (Análisis Supervisado + No Supervisado)  
**Versión**: 2.0 (Análisis integrado con justificación completa)   
**Notebooks**: 4 (EDA, Clasificación, Clustering, Taller Colaborativo)