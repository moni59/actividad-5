# actividad-5
# Actividad 5 — comparación de clasificadores con MLflow

Proyecto reproducible de clasificación binaria que compara **Regresión Logística** y **Random Forest** sobre Breast Cancer Wisconsin Diagnostic, incluido en `scikit-learn`.

## Estructura

```text
Actividad5/
├── datos/
│   ├── datos_ini/breast_cancer_wisconsin.csv
│   └── datos_limp/breast_cancer_wisconsin_limpio.csv
├── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
├── evidencias/                 # métricas, gráficos y comparación estadística
├── reportes/reporte_tecnico.pdf
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

## Fuente y diccionario

Datos de diagnóstico de cáncer de mama de Wisconsin, distribuidos como dataset de ejemplo por scikit-learn y originados en el Wisconsin Diagnostic Breast Cancer (WDBC), UCI Machine Learning Repository. 569 observaciones, 30 predictores numéricos y una etiqueta binaria. Cada predictor describe la media, error estándar o peor valor de características de núcleos celulares. Ver `sklearn.datasets.load_breast_cancer().feature_names` para los 30 nombres y sus grupos (`mean`, `error`, `worst`). `diagnostico=0` significa maligno y `diagnostico=1` benigno.

**Supuestos:** una fila corresponde a una muestra independiente; la tarea es didáctica, no clínica. El conjunto tiene ambas clases y no contiene faltantes. La escala de los predictores puede diferir. La clase positiva de todas las métricas binarias es benigno (1).

## Reproducción local

Requiere Python 3.10+.

```bash
cd Actividad5
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python fuentes/datos_prep.py
python fuentes/train.py
mlflow ui --backend-store-uri ./mlruns
```

Abrir `http://127.0.0.1:5000`. El script fija semilla 42, reserva 20% para prueba estratificada y optimiza ROC-AUC mediante `GridSearchCV` con validación cruzada estratificada de 5 folds. Imputación mediana y escalado se ajustan dentro de cada pipeline/fold. Random Forest conserva el escalado dentro del pipeline por simetría; el escalado no es necesario para ese estimador.

## Métricas y análisis

Se registran accuracy, precision, recall, F1 y ROC-AUC en prueba; además, promedio y desviación estándar de ROC-AUC en CV, hiperparámetros óptimos, modelo serializado, reporte por clase y matriz de confusión. `evidencias/` contiene tabla comparativa, ROC, matrices de confusión, ROC-AUC por fold y una prueba t pareada exploratoria. Con cinco folds la prueba estadística es poco potente y no prueba superioridad generalizable.

El script crea un run de MLflow por modelo y un run de comparación final. El almacén local se crea en `mlruns/` (no se recomienda versionar ese directorio; los resúmenes compactos están en `evidencias/`). Para usar un tracking server, cambia `set_tracking_uri` en `train.py` por la URI del servidor. Las métricas obtenidas y advertencia sobre la captura real están en `evidencias/README.md`.

## Versionamiento

Los CSV están versionados para que el repositorio sea autónomo. El CSV original se reconstruye desde el dataset empaquetado en scikit-learn con `datos_prep.py`; el CSV limpio se regenera con las mismas reglas. Los cambios se anotan en `CHANGELOG.md`. Se recomienda versionar dependencias exactas en un entorno de producción.

## Resultados reproducidos

Regresión Logística: accuracy 0.9825, ROC-AUC 0.9954. Random Forest: accuracy 0.9474, ROC-AUC 0.9937. La t pareada exploratoria de cinco folds dio p=0.285. La tabla completa, figuras y limitaciones de evidencia MLflow están en `evidencias/README.md`.

## Limitaciones y deuda técnica

El set es pequeño y se usa una única partición de prueba; no hay calibración, análisis de umbral, monitoreo de drift, validación externa ni explicabilidad formal. La prueba t con folds solapados es orientativa. No usar métricas ni modelos para decisiones médicas. MLOps futuro: tracking remoto con permisos, registro/promoción de modelos, pruebas de datos, CI, firma del dataset, monitoreo de calidad/drift, alertas y control de acceso.

## Publicar en GitHub

```bash
git add .
git commit -m "Agrega comparacion de clasificadores y seguimiento MLflow"
git branch -M main
git remote add origin https://github.com/USUARIO/Actividad5.git
git push -u origin main
```
