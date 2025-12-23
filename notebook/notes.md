## Proyecto: Clasificación de Sentimiento en Reseñas de IMDb (Film Junky Union)

**Objetivo**
Film Junky Union quiere un sistema automático para **detectar reseñas negativas** de películas. Usaremos reseñas de IMDb ya etiquetadas para entrenar y comparar varios modelos que clasifiquen cada reseña como **positiva (1)** o **negativa (0)**. El requisito del proyecto es lograr **F1 ≥ 0.85** en el conjunto de prueba.

**Datos**
El archivo `imdb_reviews.tsv` incluye:

* `review`: texto de la reseña (feature principal)
* `pos`: objetivo (`0` = negativo, `1` = positivo)
* `ds_part`: partición ya definida (`train` / `test`)
* metadatos extra (títulos, año, géneros, ratings, votos, etc.) que usaremos **solo para EDA** (no para entrenar el modelo principal, para evitar fuga de información)

**Enfoque (alto nivel)**

1. Cargar e inspeccionar los datos y validar que todo sea consistente.
2. Hacer EDA, especialmente el desbalance de clases y la longitud del texto.
3. Construir baselines fuertes con **TF-IDF + Regresión Logística** y un modelo de **boosting** bien ajustado.
4. Afinar **BERT** usando GPU como benchmark moderno.
5. Evaluar en test y analizar diferencias con reseñas escritas manualmente.

**Métrica**
Principal: **F1-score** (accuracy como secundaria).
Usamos F1 porque equilibra precisión y recall, y es más robusta que accuracy si hay desbalance.

---

## Flujo de trabajo ideal (estilo DS senior)

### 1) Setup + reglas (antes de modelos)

* Fijar `SEED` para reproducibilidad.
* Definir rutas y constantes.
* Decidir features “permitidas”:

  * **Para modelos**: usar **solo** `review`.
  * **Para EDA**: `rating`, `average_rating`, `votes`, etc. (pero **no** entrenar con ellas para evitar leakage).

Entregable: una celda de configuración clara + reglas del juego.

---

### 2) Carga de datos + sanity checks (para no sufrir después)

* Cargar TSV y confirmar filas/columnas.
* Revisar nulos en `review`, `pos`, `ds_part`.
* Confirmar que `ds_part` solo tenga `train` y `test`.
* Confirmar que `pos` sea binaria y su distribución sea coherente.

Entregable: sección corta de “integridad de datos”.

---

### 3) EDA (solo lo necesario para tomar decisiones)

Enfócate en 4 cosas:

1. **Balance de clases**: distribución de `pos` en train/test.
2. **Longitud del texto**: chars/tokens, outliers.
3. **Ejemplos**: algunas reseñas por clase (para ver HTML/ruido).
4. **Historia opcional**: relación de `pos` con `genres` o `average_rating` (EDA only).

Entregable: 2–4 gráficos + conclusiones breves.

---

### 4) Estrategia de preprocesamiento (mínima y consistente)

Crear una función tipo `clean_text()`:

* minúsculas
* remover HTML si existe
* normalizar espacios
* no destruir negaciones (evitar “limpiar demasiado”)

Entregable: mostrar antes/después en 2–3 ejemplos.

---

### 5) División correcta (evaluación limpia)

Usar `ds_part` para el split final:

* Pool de entrenamiento = `ds_part == 'train'`
* Test = `ds_part == 'test'`

Dentro del pool de entrenamiento:

* Split **train/valid** con estratificación.

Entregable: `X_train`, `X_valid`, `y_train`, `y_valid` + shapes + balance.

---

### 6) Modelo 1 — Baseline (Pipeline + GridSearchCV)

**Pipeline:** `TfidfVectorizer → LogisticRegression`
**GridSearchCV scoring:** `f1`
Grid pequeño pero inteligente:

* `ngram_range`
* `max_features`
* `min_df`
* `C`
* `class_weight`

Entregable: mejores hiperparámetros + F1 en valid + comentario.

---

### 7) Modelo 2 — Boosting pro (recomendación: XGBoost)

Pipeline también:

* `TfidfVectorizer → XGBClassifier`

Afinar de forma ligera (grid pequeño o randomized):

* `n_estimators`, `max_depth`, `learning_rate`, subsampling
  Si aplica: early stopping con validación.

Entregable: F1 en valid + comparación vs Logistic.

---

### 8) Modelo 3 — BERT (fine-tuning con GPU)

Sección aparte (es otro ecosistema):

* Tokenizer + clase Dataset + DataLoaders
* Loop de entrenamiento con optimizer + scheduler
* Validación por epoch
* Guardar el mejor checkpoint por **F1 en valid**

Entregable: curvas de loss + F1 valid + F1 final en test.

---

### 9) Evaluación final en test (una sola vez, comparación justa)

Para cada modelo:

* entrenar con la mejor config (idealmente usando todo el train pool)
* evaluar en `ds_part == 'test'`
* reportar: F1, accuracy, matriz de confusión

Entregable: tabla comparativa de resultados.

---

### 10) Reseñas manuales + análisis de desacuerdos (toque pro)

* Escribir 8–12 reseñas propias:

  * claras, mezcladas, con negaciones, etc.
* Correr los 3 modelos
* Explicar por qué difieren (TF-IDF keywords vs BERT contexto)

Entregable: mini tabla + insights.

---

### 11) Conclusiones (limpias y directas)

* Qué modelo es mejor y por qué (performance + costo).
* Confirmar si F1 ≥ 0.85.
* Qué mejorarías después (más tuning, calibración, etc.).

Con este flujo cumples el rubric y tu notebook se lee como entrega real de DS.
