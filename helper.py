# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

import plots as p   # нужно будет сделать функции plot_regression_results / print_regression_report


def divide_data(data: pd.DataFrame, target_column: str):
    """Разделение на признаки и таргет."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


# ========= МЕТРИКИ РЕГРЕССИИ ========= #

def calculate_regression_metrics(y_true, y_pred):
    """
    Считаем метрики регрессии.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }
    return metrics


def evaluate_regression(y_true, y_pred, model_name: str = "Model", enable_plot: bool = True):
    """
    Обёртка над calculate_regression_metrics + вызов функций визуализации.
    """
    metrics = calculate_regression_metrics(y_true, y_pred)

    if enable_plot:
        # Эти функции нужно реализовать в plots.py
        # например: график распределения ошибок, y_true vs y_pred и т.п.
        p.plot_regression_results(metrics, model_name)
        p.print_regression_report(metrics, model_name)

    return metrics


# ========= ОБУЧЕНИЕ ОДНОЙ МОДЕЛИ (train / test) ========= #

def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, seed=None):
    """
    Обучение одной регрессионной модели и оценка качества на тесте.
    """
    # фиксируем сид
    if seed is not None:
        if hasattr(model, "random_state"):
            model.set_params(random_state=seed)
        if hasattr(model, "seed"):
            model.set_params(seed=seed)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_regression(
        y_true=y_test,
        y_pred=y_pred,
        model_name=model_name,
        enable_plot=False,
    )

    return metrics


# ========= ОБУЧЕНИЕ С CV ========= #

def train_evaluate_model_cv(
    model,
    model_name,
    X,
    y,
    preprocessor=None,
    cv: int = 5,
    seed: int | None = None,
):
    """
    Обучение регрессионной модели с кросс-валидацией и (опциональным) препроцессором.
    """
    # сид
    if seed is not None:
        if hasattr(model, "random_state"):
            model.set_params(random_state=seed)
        if hasattr(model, "seed"):
            model.set_params(seed=seed)

    # конструируем пайплайн
    if isinstance(preprocessor, Pipeline):
        preprocessor.steps.append(("model", model))
        pipeline = preprocessor
    elif preprocessor is not None:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
    else:
        pipeline = model

    # метрики для cross_validate
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    mae = -cv_results["test_mae"].mean()
    mse = -cv_results["test_mse"].mean()
    rmse = np.sqrt(mse)
    r2 = cv_results["test_r2"].mean()

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

    # тут можно отрисовать что-то типа barplot по метрикам
    p.plot_regression_results(metrics, model_name)

    return metrics


def train_evaluate_models_cv(models: list, X, y, preprocessor=None, cv=5, seed=None):
    """
    Запуск нескольких регрессионных моделей с CV и сравнение по метрикам.
    models: list of (model_name, model_instance)
    """
    all_metrics = {}

    for model_name, model in models:
        current_model = clone(model)
        current_preprocessor = clone(preprocessor)

        all_metrics[model_name] = train_evaluate_model_cv(
            current_model,
            model_name,
            X,
            y,
            current_preprocessor,
            cv,
            seed
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")

    plt.figure(figsize=(10, 4))
    sns.heatmap(metrics_df, cmap="RdBu_r", annot=True, fmt=".3f")
    plt.title("Model Evaluation Metrics Comparison (Regression)")
    plt.tight_layout()
    plt.show()

    return metrics_df


def train_evaluate_models(models: list, X_train, y_train, X_test, y_test, seed=None):
    """
    Обучение нескольких регрессионных моделей на train/test
    и сравнение по метрикам.
    """
    all_metrics = {}

    for model_name, model in models:
        current_model = clone(model)

        all_metrics[model_name] = train_evaluate_model(
            current_model,
            model_name,
            X_train,
            y_train,
            X_test,
            y_test,
            seed,
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient="index")

    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap="RdBu_r", annot=True, fmt=".3f")
    plt.title("Model Evaluation Metrics Comparison (Regression)")
    plt.tight_layout()
    plt.show()

    return metrics_df


# ========= ОБРАБОТКА ВЫБРОСОВ ========= #

def winsorize_outliers(df: pd.DataFrame, column_name: str,
                       lower_bound: float | None = None,
                       upper_bound: float | None = None) -> pd.DataFrame:
    """
    Обрезание выбросов (winsorization) для числового признака.
    """
    df = df.copy()

    if lower_bound is not None:
        df.loc[df[column_name] < lower_bound, column_name] = lower_bound
    if upper_bound is not None:
        df.loc[df[column_name] > upper_bound, column_name] = upper_bound

    return df
