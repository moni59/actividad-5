# Changelog

## 1.0.0 — 2026-10-18
- Se preparó el dataset WDBC (569 filas, 30 variables) en CSV original y limpio.
- Se implementaron pipelines de Regresión Logística y Random Forest con búsqueda de hiperparámetros y CV estratificada de 5 folds.
- Se registran parámetros, métricas y artefactos por run en MLflow local.
- Se agregaron métricas estandarizadas, ROC, matrices de confusión y comparación pareada exploratoria.
- Resultados reproducidos con semilla fija; ver `evidencias/` y `reportes/reporte_tecnico.pdf`.
