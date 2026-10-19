# Evidencias

Los CSV y PNG son resultados del split estratificado 80/20 (random_state=42) y GridSearchCV de 5 folds descritos en el reporte.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC prueba | ROC-AUC CV (media ± DE) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 | 0.9959 ± 0.0050 |
| Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9937 | 0.9925 ± 0.0050 |

La comparación pareada de ROC-AUC de los 5 folds obtuvo t=1.233, p=0.285, diferencia media=0.0034 (Logistic Regression − Random Forest). Interprétese con cautela por el tamaño pequeño de CV.

`exploracion_datos.png`, `curvas_roc.png` y las matrices de confusión son las visualizaciones generadas. El entorno de preparación no permitió instalar MLflow, por lo que esta carpeta no contiene una captura real del panel ni un almacén `mlruns`. El script `fuentes/train.py` registra los runs cuando se instala `requirements.txt`. Tras ejecutarlo, abre la UI y toma captura de Experiments / Runs para adjuntarla como evidencia de la ejecución real:

```bash
mlflow ui --backend-store-uri ./mlruns
```
