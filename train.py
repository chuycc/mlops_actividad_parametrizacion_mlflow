"""
Operaciones de aprendizaje automático (Gpo 10)
Actividad Individual | Código de experimentación base con MLFlow

Alumno:
    Jesús Carrillo Cantú - A01840421

Dataset:
    IBM HR Analytics Employee Attrition & Performance.
    https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Uso:
    python train.py
    python train.py --data-path data/WA_Fn-UseC_-HR-Employee-Attrition.csv
    python train.py --C 1 --class-weight balanced --drop-features Gender HourlyRate MonthlyRate
"""

import argparse
import hashlib
import json
import random
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # backend sin interfaz gráfica, para correr desde la terminal
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

# Origen de los datos
KAGGLE_DATASET = 'pavansubhasht/ibm-hr-analytics-attrition-dataset/versions/1'
KAGGLE_CSV_FILENAME = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

# Variable objetivo. 1 = el empleado abandonó la empresa
TARGET = 'Attrition'
TARGET_MAPPING = {'No': 0, 'Yes': 1}
CLASS_LABELS = ['No abandonó', 'Abandonó']

# Columnas sin valor para el modelo
USELESS_COLUMNS = ['Over18', 'StandardHours', 'EmployeeCount', 'EmployeeNumber']

NUMERIC_FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
    'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager',
]

ORDINAL_FEATURES = {
    'Education': [1, 2, 3, 4, 5],
    'EnvironmentSatisfaction': [1, 2, 3, 4],
    'JobInvolvement': [1, 2, 3, 4],
    'JobLevel': [1, 2, 3, 4, 5],
    'JobSatisfaction': [1, 2, 3, 4],
    'PerformanceRating': [3, 4],
    'RelationshipSatisfaction': [1, 2, 3, 4],
    'StockOptionLevel': [0, 1, 2, 3],
    'WorkLifeBalance': [1, 2, 3, 4],
}

BINARY_FEATURES = ['Gender', 'OverTime']

NOMINAL_FEATURES = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']

ALL_FEATURES = NUMERIC_FEATURES + list(ORDINAL_FEATURES) + BINARY_FEATURES + NOMINAL_FEATURES

CV_SCORING = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': 'recall',
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Entrena una regresión logística L2 (RIDGE) y registra la corrida en MLflow.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hiperparámetros del modelo
    model = parser.add_argument_group('hiperparámetros del modelo')
    model.add_argument('--C', type=float, default=10.0,
                       help='Inverso de la fuerza de regularización L2 (menor C = más regularización).')
    model.add_argument('--class-weight', choices=['none', 'balanced'], default='none',
                       help='Pesos por clase.')
    model.add_argument('--solver', choices=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                       default='lbfgs', help='Algoritmo de optimización.')
    model.add_argument('--max-iter', type=int, default=1000,
                       help='Número máximo de iteraciones del solver.')
    model.add_argument('--tol', type=float, default=1e-4,
                       help='Tolerancia del criterio de paro del solver.')

    # Parámetros de datos y evaluación
    data = parser.add_argument_group('datos y evaluación')
    data.add_argument('--data-path', default=None,
                      help='Ruta al dataset local. Si se omite, descarga el dataset de Kaggle.')
    data.add_argument('--drop-features', nargs='*', default=[], metavar='COL',
                      help='Variables a eliminar.')
    data.add_argument('--test-size', type=float, default=0.15,
                      help='Proporción del conjunto de test.')
    data.add_argument('--cv-splits', type=int, default=5, help='Folds de la validación cruzada.')
    data.add_argument('--cv-repeats', type=int, default=3, help='Repeticiones de la validación cruzada.')
    data.add_argument('--seed', type=int, default=17,
                      help='Semilla para reproducibilidad.')

    # Parámetros de MLflow
    tracking = parser.add_argument_group('MLflow')
    tracking.add_argument('--tracking-uri', default='sqlite:///mlflow.db', help='Tracking URI de MLflow.')
    tracking.add_argument('--experiment-name', default='ibm-hr-attrition-ridge-baseline',
                          help='Nombre del experimento de MLflow.')
    tracking.add_argument('--run-name', default=None, help='Nombre de la corrida.')

    args = parser.parse_args(argv)

    # Validar argumentos
    unknown = sorted(set(args.drop_features) - set(ALL_FEATURES))
    if unknown:
        parser.error(f'Estas columnas a eliminar no existen en el dataset: {unknown}')
    if not 0 < args.test_size < 0.5:
        parser.error('--test-size debe estar entre 0 y 0.5')
    if args.cv_splits < 2 or args.cv_repeats < 1:
        parser.error('--cv-splits debe ser al menos 2 y --cv-repeats al menos 1')
    if args.data_path and not Path(args.data_path).is_file():
        parser.error(f'No se encuentra el dataset: {args.data_path}')

    args.drop_features = sorted(set(args.drop_features))

    return args


def run_name(args):
    if args.run_name:
        return args.run_name
    return f'ridge_C{args.C:g}_cw-{args.class_weight}_drop{len(args.drop_features)}'


def download_dataset():
    """Descarga el dataset de Kaggle."""
    try:
        import kagglehub
    except ImportError:
        raise SystemExit('Falta la dependencia kagglehub. También puedes usar --data-path con un CSV local.') from None

    try:
        dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        raise SystemExit(
            f'No se pudo descargar {KAGGLE_DATASET} de Kaggle: {exc}. Usa --data-path con un CSV local.'
        ) from None

    dataset_path = dataset_dir / KAGGLE_CSV_FILENAME
    if not dataset_path.is_file():
        raise SystemExit(f'No se encuentra el dataset de Kaggle {KAGGLE_CSV_FILENAME} en {dataset_dir}')
    return dataset_path


def resolve_data_path(data_path):
    """Regresa la ruta del dataset."""
    if data_path:
        return Path(data_path)
    return download_dataset()


def file_sha256(path):
    """Huella del archivo de datos, para saber si las corridas usaron el mismo dataset."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_features(drop_features):
    return [col for col in ALL_FEATURES if col not in drop_features]


def load_data(path, features):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df.drop(columns=USELESS_COLUMNS)

    X = df[features]
    y = df[TARGET].map(TARGET_MAPPING).astype(int)
    return X, y


def split_data(X, y, test_size, seed):
    """Partición estratificada 70/15/15"""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=2 * test_size, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    return X_trainval, X_test, y_trainval, y_test


def build_preprocessor(features):
    numeric = [col for col in NUMERIC_FEATURES if col in features]
    ordinal = [col for col in ORDINAL_FEATURES if col in features]
    binary = [col for col in BINARY_FEATURES if col in features]
    nominal = [col for col in NOMINAL_FEATURES if col in features]

    ordinal_pipeline = Pipeline([
        ('ordinal', OrdinalEncoder(categories=[ORDINAL_FEATURES[col] for col in ordinal],
                                   handle_unknown='use_encoded_value',
                                   unknown_value=-1)),
        ('scaler', MinMaxScaler()),
    ])

    transformers = [
        ('num', MinMaxScaler(), numeric),
        ('catOrd', ordinal_pipeline, ordinal),
        ('catBin', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), binary),
        ('catNom', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nominal),
    ]

    # Quitar los grupos que quedaron vacíos
    transformers = [t for t in transformers if t[2]]

    return ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)


def build_model(args):
    """Regresión logística con regularización L2 (RIDGE)."""
    return LogisticRegression(
        C=args.C,
        class_weight=None if args.class_weight == 'none' else args.class_weight,
        solver=args.solver,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
    )


def build_pipeline(args, features):
    return Pipeline(steps=[('ct', build_preprocessor(features)), ('m', build_model(args))])


def build_model_signature(pipeline, X):
    """Firma del modelo con los tipos del dataset. Esto es para evitar los warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Hint: Inferred schema contains integer column')
        return infer_signature(X, pipeline.predict(X))


def cross_validate_model(pipeline, X, y, n_splits, n_repeats, seed):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = cross_validate(pipeline, X, y, scoring=CV_SCORING, cv=cv, n_jobs=1)

    metrics = {}
    for name in CV_SCORING:
        values = scores[f'test_{name}']
        metrics[f'cv_{name}_mean'] = float(values.mean())
        metrics[f'cv_{name}_std'] = float(values.std())
    return metrics


def evaluate_on_test(pipeline, X_test, y_test):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred, zero_division=0),
        'test_roc_auc': roc_auc_score(y_test, y_proba),
        'test_pr_auc': average_precision_score(y_test, y_proba),
    }
    return {name: float(value) for name, value in metrics.items()}, y_pred


def save_confusion_matrix(y_true, y_pred, path, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=CLASS_LABELS,
                                            cmap='Blues', ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(experiment_name, metrics):
    print(f'Experimento: {experiment_name}')
    for name, value in sorted(metrics.items()):
        print(f'  {name:<20} {value:.4f}' if isinstance(value, float) else f'  {name:<20} {value}')
    print(f'Artefactos: {mlflow.get_artifact_uri()}')


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = resolve_data_path(args.data_path)
    features = selected_features(args.drop_features)
    X, y = load_data(data_path, features)
    X_trainval, X_test, y_trainval, y_test = split_data(X, y, args.test_size, args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name(args)):
        params = {
            # Hiperparámetros del modelo
            'model_type': 'LogisticRegression_L2',
            'C': args.C,
            'class_weight': args.class_weight,
            'solver': args.solver,
            'max_iter': args.max_iter,
            'tol': args.tol,

            # Datos, evaluación y reproducibilidad
            'drop_features': ', '.join(args.drop_features) or 'none',
            'n_features': len(features),
            'test_size': args.test_size,
            'cv_splits': args.cv_splits,
            'cv_repeats': args.cv_repeats,
            'seed': args.seed,
            'n_train_val': len(X_trainval),
            'n_test': len(X_test),
        }

        for name, value in params.items():
            mlflow.log_param(name, value)
        mlflow.set_tag('data_sha256', file_sha256(data_path))

        # Validación cruzada
        pipeline = build_pipeline(args, features)
        cv_metrics = cross_validate_model(pipeline, X_trainval, y_trainval,
                                          args.cv_splits, args.cv_repeats, args.seed)

        # Entrenamiento
        pipeline.fit(X_trainval, y_trainval)
        test_metrics, y_pred = evaluate_on_test(pipeline, X_test, y_test)

        metrics = {**cv_metrics, **test_metrics, 'n_iter': int(np.max(pipeline.named_steps['m'].n_iter_))}
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Artefactos
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = Path(tmp_dir) / 'confusion_matrix_test.png'
            metrics_path = Path(tmp_dir) / 'metrics.json'

            save_confusion_matrix(y_test, y_pred, cm_path, 'Matriz de confusión (test)')
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(metrics_path))

        # Modelo entrenado
        input_example = X_test.head(3)
        mlflow.sklearn.log_model(
            pipeline,
            name='model',
            signature=build_model_signature(pipeline, input_example),
            input_example=input_example,
            skops_trusted_types=['sklearn.compose._column_transformer._RemainderColsList'],
        )

        print_summary(args.experiment_name, metrics)


if __name__ == '__main__':
    main()