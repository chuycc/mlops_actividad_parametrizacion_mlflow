# Código de experimentación base con MLflow

Operaciones de aprendizaje automático (Gpo 10)

Jesús Carrillo Cantú - A01840421

Modelo base: regresión logística con regularización L2 (RIDGE) sobre el dataset
[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
Clasificación binaria de la variable objetivo `Attrition` (1 = el empleado abandonó la empresa).

## Estructura

| Archivo | Contenido |
|---|---|
| `train.py` | Carga de datos, preprocesamiento, validación cruzada, entrenamiento, evaluación y registro en MLflow |
| `run_experiments.sh` | Las corridas del experimento |
| `requirements.txt` | Dependencias con versiones fijas |
| `notebooks/EDA.ipynb` | Exploración de datos (el entrenamiento se migró a `train.py`) |

## Instalación

Requiere Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

El script puede obtener el dataset de dos maneras:

1. **Descarga automática (por defecto):** si no se pasa `--data-path`, se descarga la versión 1 del dataset [IBM HR Analytics Employee Attrition & Performance de Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
2. **Archivo local:** `--data-path` usa un CSV local y no descarga nada.

```bash
python train.py                                                          # descarga de Kaggle
python train.py --data-path data/WA_Fn-UseC_-HR-Employee-Attrition.csv   # CSV local
```

Cada corrida registra el tag `data_sha256` con la huella del CSV para confirmar reproducibilidad con el miso dataset.

## Ejecución

```bash
python train.py                      # valores por defecto (baseline RIDGE, C=10)
python train.py --C 1 --class-weight balanced --drop-features Gender HourlyRate MonthlyRate
python train.py --help

mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://127.0.0.1:5000
```

## Argumentos

**Hiperparámetros del modelo**

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | 10 | Inverso de la fuerza de regularización L2 (menor C = más regularización) |
| `--class-weight` | `none` | Pesos por clase: `none` o `balanced` |
| `--solver` | `lbfgs` | Algoritmo de optimización |
| `--max-iter` | 1000 | Número máximo de iteraciones del solver |
| `--tol` | 1e-4 | Tolerancia del criterio de paro |

**Datos y evaluación**

| Argumento | Default | Descripción |
|---|---|---|
| `--data-path` | (descarga de Kaggle) | Ruta a un CSV local |
| `--drop-features` | (ninguna) | Variables a eliminar, separadas por espacio |
| `--test-size` | 0.15 | Proporción de test. Validation usa la misma, por eso debe ser menor a 0.5 |
| `--cv-splits` | 5 | Folds de la validación cruzada |
| `--cv-repeats` | 3 | Repeticiones de la validación cruzada |
| `--seed` | 17 | Semilla aleatoria |

**MLflow**

| Argumento | Default | Descripción |
|---|---|---|
| `--tracking-uri` | `sqlite:///mlflow.db` | Tracking URI |
| `--experiment-name` | `ibm-hr-attrition-ridge-baseline` | Nombre del experimento |
| `--run-name` | `ridge_C<C>_cw-<class_weight>_drop<n>` | Nombre de la corrida |

## Flujo del entrenamiento

1. **Carga:** elimina las columnas sin valor (`EmployeeNumber`, `Over18`, `StandardHours`, `EmployeeCount`) y las de `--drop-features`.
2. **Partición estratificada:** train / validation / test (70/15/15 por defecto).
3. **Validación cruzada:** `RepeatedStratifiedKFold` (5 x 3 por defecto) sobre train + validation.
4. **Entrenamiento final:** con todo train + validation.
5. **Evaluación en test:** documenta el desempeño del modelo final.

Preprocesamiento por tipo de variable:
- `MinMaxScaler` para numéricas
- `OrdinalEncoder` con categorías explícitas para ordinales
- `OneHotEncoder` para binarias y nominales

## Corridas

| Tipo | Contenido |
|---|---|
| Parámetros (`log_param`) | `model_type`, `C`, `class_weight`, `solver`, `max_iter`, `tol`, `drop_features`, `n_features`, `test_size`, `cv_splits`, `cv_repeats`, `seed`, `n_train_val`, `n_test` |
| Métricas (`log_metric`) | Media y desviación estándar en validación cruzada (`cv_*_mean`, `cv_*_std`) y métricas en test (`test_*`) de accuracy, precision, recall, F1, ROC-AUC y PR-AUC; además `n_iter` |
| Artefactos (`log_artifact`) | `confusion_matrix_test.png` y `metrics.json` |
| Modelo (`mlflow.sklearn.log_model`) | Pipeline completo (preprocesamiento + modelo) con firma y ejemplo de entrada |
| Tag | `data_sha256`: huella del CSV usado |


## Reproducibilidad

**Semilla:** el argumento `--seed` (17 por defecto) controla todas las fuentes de aleatoriedad:

1. `random.seed` y `np.random.seed`.
2. `random_state` de las dos particiones estratificadas.
3. `random_state` de `RepeatedStratifiedKFold`.
4. `random_state` de `LogisticRegression`.

**Datos:** la versión del dataset en Kaggle está fija (`versions/1`) y el tag `data_sha256` permite verificar que todas las corridas usaron el mismo archivo.

**Dependencias:** `requirements.txt` fija las versiones exactas con las que se ejecutó el experimento.