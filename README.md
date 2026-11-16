<img width="1280" height="640" alt="kappa_abandono-bancario" src="https://github.com/user-attachments/assets/a9d3a9ad-4dcd-4aa8-a16e-3b78ce826c25" />
# Κ Proyecto Kappa | Predicción de Abandono de Clientes en Beta Bank

## Descripción

Beta Bank identificó un problema de negocio relevante: cada mes una parte de sus clientes abandona la entidad financiera. Debido a que retener clientes existentes suele ser más rentable que captar nuevos usuarios, el banco busca desarrollar un sistema capaz de anticipar qué clientes presentan riesgo de abandono.

En este proyecto se construyeron y evaluaron modelos de Machine Learning para predecir la cancelación de clientes utilizando información demográfica, financiera y de comportamiento bancario. El objetivo principal fue maximizar el valor de la métrica F1 y desarrollar un modelo capaz de identificar correctamente a los clientes con mayor probabilidad de abandonar el banco. 【1-dd1c83】

---

## Objetivos

- Analizar la calidad y estructura del conjunto de datos.
- Preparar las variables para el entrenamiento de modelos predictivos.
- Estudiar el desequilibrio de clases de la variable objetivo.
- Entrenar distintos modelos de clasificación.
- Comparar el desempeño utilizando F1 Score y ROC-AUC.
- Aplicar técnicas de balanceo de clases.
- Optimizar hiperparámetros para maximizar el desempeño.
- Seleccionar el modelo más robusto.
- Evaluar el modelo final sobre datos no utilizados durante el entrenamiento. 【1-dd1c83】

---

## Herramientas utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Jupyter Notebook

Modelos evaluados:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Técnicas de balanceo:

- Oversampling
- Undersampling 【1-dd1c83】

---

## Fuente de datos

Archivo utilizado:

```text
Churn.csv
```

### Información del cliente

- `CustomerId`
- `Surname`
- `Age`
- `Gender`
- `Geography`

### Información financiera

- `CreditScore`
- `Balance`
- `EstimatedSalary`

### Relación con el banco

- `Tenure`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`

### Variable objetivo

- `Exited`
  - 1 = el cliente abandonó el banco.
  - 0 = el cliente permanece en el banco. 【1-dd1c83】

---

## Metodología

### 1. Exploración y limpieza de datos

El dataset contiene:

- 10.000 registros.
- 14 variables.

Durante la revisión inicial se identificó que la única columna con valores ausentes era:

```text
Tenure
```

Se analizaron sus estadísticas descriptivas y se observó que la media y la mediana eran prácticamente iguales (≈ 5 años), por lo que los valores faltantes fueron reemplazados utilizando la mediana para minimizar la introducción de sesgos artificiales. 【1-dd1c83】

---

### 2. Análisis del equilibrio de clases

La variable objetivo presentó un claro desbalance:

| Clase | Significado | Proporción |
|---------|---------|---------:|
| 0 | Cliente permanece | 79.63% |
| 1 | Cliente abandona | 20.37% |

Este desequilibrio representa un desafío importante para los modelos de clasificación, ya que podrían favorecer excesivamente la clase mayoritaria. 【1-dd1c83】

---

### 3. Preparación de variables

Se eliminaron variables que no aportaban valor predictivo:

```text
CustomerId
RowNumber
Surname
```

Posteriormente se aplicó codificación One-Hot Encoding sobre:

- Geography
- Gender

utilizando:

```python
drop_first=True
```

para evitar redundancia entre variables. 【1-dd1c83】

---

### 4. División de los datos

Los datos fueron divididos en:

- Entrenamiento: 60%
- Validación: 20%
- Prueba: 20%

Esta estructura permitió entrenar modelos, ajustar hiperparámetros y reservar información independiente para la evaluación final. 【1-dd1c83】

---

### 5. Escalado de características

Para los modelos sensibles a la escala de los datos se aplicó:

```text
StandardScaler
```

sobre las variables numéricas, garantizando una comparación adecuada entre atributos de distintas magnitudes. 【1-dd1c83】

---

## Modelos evaluados

### 1. Regresión logística

Resultado en validación:

| Métrica | Valor |
|----------|--------:|
| F1 Score | 0.331 |
| ROC-AUC | 0.759 |

El modelo mostró una capacidad razonable para diferenciar clases según ROC-AUC, pero presentó dificultades para identificar correctamente la clase positiva (clientes que abandonan el banco). 【1-dd1c83】

---

### 2. Árbol de decisión

Mejor configuración:

```text
max_depth = 6
```

Resultados:

| Métrica | Valor |
|----------|--------:|
| F1 Score | 0.570 |
| ROC-AUC | 0.684 |

El árbol mejoró significativamente el valor F1 respecto a la regresión logística, acercándose al objetivo planteado por el proyecto. 【1-dd1c83】

---

### 3. Bosque aleatorio

Mejor configuración encontrada:

```text
RandomForestClassifier(
    max_depth=15,
    n_estimators=300
)
```

Resultados:

| Métrica | Valor |
|----------|--------:|
| F1 Score | 0.589 |
| ROC-AUC | 0.844 |

Fue el modelo con mejor desempeño general antes de aplicar técnicas de balanceo. 【1-dd1c83】

---

## Técnicas de balanceo de clases

### Oversampling

Se aplicó sobremuestreo sobre la clase minoritaria utilizando:

```python
resample()
```

Resultados:

| Métrica | Valor |
|----------|--------:|
| F1 Score | 0.613 |
| ROC-AUC | 0.843 |

El sobremuestreo logró incrementar la capacidad del modelo para detectar clientes que abandonan el banco, permitiendo superar el umbral mínimo de F1 requerido por el proyecto. 【1-dd1c83】

---

### Undersampling

Se aplicó submuestreo reduciendo la cantidad de observaciones pertenecientes a la clase mayoritaria.

Resultados:

| Métrica | Valor |
|----------|--------:|
| F1 Score | 0.582 |
| ROC-AUC | 0.846 |

Aunque obtuvo el mejor valor ROC-AUC, su F1 resultó inferior al conseguido mediante sobremuestreo, por lo que no fue seleccionado como modelo final. 【1-dd1c83】

---

## Comparación de modelos

| Modelo | F1 Score | ROC-AUC |
|----------|--------:|--------:|
| Regresión logística | 0.331 | 0.759 |
| Árbol de decisión | 0.570 | 0.684 |
| Bosque aleatorio | 0.589 | 0.844 |
| Bosque aleatorio + Oversampling | 0.613 | 0.843 |
| Bosque aleatorio + Undersampling | 0.582 | 0.846 |

【1-dd1c83】

---

## Modelo seleccionado

### Random Forest + Oversampling

Configuración:

```text
max_depth = 15
n_estimators = 300
```

Razones de selección:

- Mejor F1 Score.
- Superó el requisito mínimo del proyecto.
- Mantuvo un valor ROC-AUC elevado.
- Mostró un desempeño estable entre validación y prueba. 【1-dd1c83】

---

## Evaluación final

Resultados sobre el conjunto de prueba:

| Métrica | Valor |
|----------|--------:|
| F1 Score | 0.601 |
| ROC-AUC | 0.855 |

El modelo supera exitosamente el requisito del proyecto:

```text
F1 ≥ 0.59
```

y demuestra una buena capacidad para distinguir entre clientes que abandonan el banco y clientes que permanecen activos. 【1-dd1c83】

---

## Visualizaciones desarrolladas

- Distribución de clases de la variable objetivo.
- Comparación del desequilibrio y equilibrio de clases.
- Curva ROC para Regresión Logística.
- Curva ROC para Árbol de Decisión.
- Curva ROC para Bosque Aleatorio.
- Curva ROC para Bosque Aleatorio con Oversampling.
- Curva ROC para Bosque Aleatorio con Undersampling.

【1-dd1c83】

---

## Archivos principales

- `kappa_bank_customer_churn_prediction.ipynb`
- `Churn.csv`

---

## Conclusión

El proyecto permitió desarrollar un modelo capaz de predecir qué clientes presentan riesgo de abandonar Beta Bank utilizando información histórica sobre su comportamiento y características financieras.

Tras evaluar distintos algoritmos de clasificación, el modelo que obtuvo el mejor desempeño fue un **Random Forest** combinado con una estrategia de **Oversampling** para corregir el desequilibrio de clases. Este enfoque alcanzó un **F1 Score de 0.601** sobre el conjunto de prueba y un **ROC-AUC de 0.855**, superando el requisito mínimo establecido para el proyecto. 【1-dd1c83】

Los resultados muestran que el tratamiento del desbalance de clases fue un factor determinante para mejorar la capacidad del modelo de identificar clientes que abandonan el banco. Mientras que los modelos entrenados sobre los datos originales mostraban limitaciones en la detección de la clase minoritaria, el sobremuestreo permitió incrementar significativamente el valor F1 sin sacrificar la capacidad de discriminación medida mediante ROC-AUC. 【1-dd1c83】

Desde una perspectiva de negocio, la solución desarrollada proporciona una herramienta predictiva que puede ayudar a Beta Bank a identificar clientes con alto riesgo de cancelación, permitiendo implementar campañas de retención más efectivas y reducir las pérdidas asociadas a la fuga de clientes. 【1-dd1c83】
