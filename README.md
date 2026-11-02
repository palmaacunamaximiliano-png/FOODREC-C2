# FoodRec — Recomendación secuencial de recetas

Proyecto académico que compara un recomendador secuencial basado
en Transformer (SASRec) con popularidad, reglas de asociación e
ítem–ítem mediante similitud coseno.

## Problema

Apoyar a una persona que utiliza una plataforma de recetas a descubrir
qué receta explorar después, utilizando su historial de interacciones.

El objetivo predictivo es la siguiente receta reseñada.
Una reseña no demuestra que la persona haya preparado la receta
ni que le haya gustado.

## Datos

Fuente: [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).

Archivos originales:
- `RAW_interactions.csv`
- `RAW_recipes.csv`

Los archivos originales deben descargarse por separado y colocarse
en `Data_raw/`. No se incluyen en el repositorio.

| Indicador | Original | Después de limpieza y 5-core |
|---|---:|---:|
| Interacciones | 1.132.367 | 555.618 |
| Usuarios | 226.570 | 17.813 |
| Recetas | 231.637 | 41.240 |

El filtro 5-core se aplica iterativamente hasta conservar usuarios
con al menos cinco recetas distintas y recetas con al menos cinco usuarios.

Las interacciones se ordenan por fecha. Los empates dentro de un día
se resuelven mediante un desempate pseudoaleatorio con semilla fija.
El índice 0 de recetas se reserva para relleno.

Los datos procesados exportados omiten los identificadores originales
de usuarios. Los índices internos no constituyen una garantía de anonimato.

## Estructura

| Ruta | Contenido |
|---|---|
| `Notebooks/01_entrega1.ipynb` | Experimento y salidas |
| `Data_raw/` | CSV originales; excluidos de Git |
| `data/secuencias.parquet` | Interacciones ordenadas y partición |
| `data/mapa_items.csv` | Índices y atributos de recetas |
| `data/metricas.csv` | Métricas finales de prueba |
| `modelos/sasrec.pt` | Checkpoint seleccionado |
| `modelos/popularidad.csv` | Frecuencias de entrenamiento |
| `modelos/reglas_asociacion.csv` | Reglas aprendidas |
| `modelos/items_normalizados.npz` | Matriz dispersa ítem–usuario |
| `Resultados/` | Gráficos, diagnósticos y tablas |
| `requirements.txt` | Dependencias |

## Ejecución

Entorno utilizado: Python 3.12.13, VS Code y PyTorch 2.15.0
en macOS con Apple Silicon. SASRec se entrenó con MPS;
las líneas base utilizaron CPU.

Para una instalación aislada:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

1. Descargar los CSV y colocarlos en `Data_raw/`.
2. Abrir `Notebooks/01_entrega1.ipynb`.
3. Seleccionar el kernel del entorno con las dependencias instaladas.
4. Ejecutar las celdas en orden.

El notebook utiliza MPS cuando está disponible y CPU en caso contrario.
La semilla principal es 42. Las versiones y el dispositivo pueden
introducir diferencias numéricas entre ejecuciones.

La celda de prueba recupera `Resultados/rankings_test.json` si existe.
Ese archivo corresponde al experimento final guardado: no debe mezclarse
con otros checkpoints o particiones.

## Protocolo

Partición leave-last-out por usuario:

| Conjunto | Interacciones |
|---|---:|
| Entrenamiento | 519.992 |
| Validación | 17.813 |
| Prueba | 17.813 |

Popularidad, reglas y similitudes se calculan solo con entrenamiento.
SASRec se ajusta con entrenamiento y se selecciona con validación.

Para prueba, el historial contiene entrenamiento y la interacción
de validación, sin actualizar parámetros.

La evaluación utiliza el catálogo completo y excluye recetas ya
observadas en el historial de entrada y el índice de relleno.
No se utilizan negativos muestreados en evaluación.

Métricas:
- HitRate@K.
- NDCG@K.
- MRR@K, truncado a K.
- Cobertura@K sobre las 41.240 recetas.

Se utilizan K = 5, 10 y 20. Las métricas se verifican con un
ejemplo didáctico y un usuario real.

## Modelos

### Popularidad

Ordena por número de usuarios de entrenamiento.
Excluye recetas vistas.

### Reglas de asociación

FP-Growth sobre historiales de entrenamiento considerados como canastas.
Soporte mínimo 0,005, conjuntos de hasta dos recetas y reglas con lift > 1.

Ordena por lift, confianza y soporte. Completa posiciones faltantes
con popularidad. Se obtuvieron 54 reglas.

### Ítem–ítem

Similitud coseno entre vectores binarios de usuarios por receta.
Suma similitudes con las últimas 20 recetas, con decaimiento 0,9.

### SASRec

Embeddings de receta y posición, autoatención causal, conexiones
residuales, normalización y red feed-forward.

Se compararon tres configuraciones:

| Configuración | Dimensión | Cabezas | Bloques | Dropout | Learning rate |
|---|---:|---:|---:|---:|---:|
| A | 64 | 2 | 1 | 0,2 | 0,001 |
| B | 64 | 2 | 2 | 0,2 | 0,001 |
| C | 128 | 2 | 1 | 0,3 | 0,0005 |

Entrenamiento:
- Longitud máxima: 50 posiciones.
- Lote: 128 bloques.
- Pérdida BCE con positivos y negativos muestreados.
- Optimizador Adam y recorte de gradiente.
- Máximo de 20 épocas.
- Parada temprana con paciencia de 3.
- Selección según NDCG@10 de validación.

Se seleccionó SASRec_A en la época 3:
2.692.736 parámetros y checkpoint de aproximadamente 10,78 MB.

## Resultados finales

Valores de HR, NDCG, MRR y cobertura expresados como proporciones.
Los resultados completos están en `data/metricas.csv`.

| Modelo | HR@20 | NDCG@10 | NDCG@20 | MRR@20 | Cobertura@20 |
|---|---:|---:|---:|---:|---:|
| Popularidad | 0,041150 | 0,011602 | 0,015871 | 0,008981 | 0,002061 |
| Ítem–ítem | 0,016224 | 0,005944 | 0,007285 | 0,004829 | 0,744956 |
| Reglas + popularidad | 0,041150 | 0,011528 | 0,015825 | 0,008934 | 0,002061 |
| SASRec | 0,038399 | 0,011639 | 0,015262 | 0,008925 | 0,020951 |

SASRec no demuestra superioridad general frente a popularidad.
Su mejora puntual de NDCG@10 es mínima y presenta menor HR@20,
aunque amplía la cobertura aproximadamente diez veces.

En historiales largos obtiene resultados ligeramente mejores
que popularidad. Ítem–ítem logra la mayor cobertura y mayor acierto
en objetivos de cola larga a K = 10, pero menor acierto global.

La búsqueda de las tres configuraciones de SASRec demoró aproximadamente
763 segundos, incluyendo 52 segundos de entrenamiento y 711 de validación.

## Diagnósticos y límites

- La similitud de ingredientes entre recetas consecutivas no mostró
  una ventaja concluyente frente a historiales desordenados.
- Desordenar las últimas 50 recetas redujo ligeramente el NDCG@10
  de SASRec, pero el intervalo bootstrap de la diferencia incluyó cero.
- La atención se presenta para tres usuarios seleccionados por longitud
  de historial, sin elegirlos por sus aciertos.
- La proyección PCA de 2.400 recetas explica el 32,15 % de la varianza
  de esa muestra y muestra superposición entre categorías.
- Las interacciones incluyen todas las valoraciones.
- Las fechas tienen precisión diaria.
- Leave-last-out por usuario no impone un corte temporal global.
- El filtro 5-core condiciona la población evaluada.
- La atención no es una explicación causal.
- Los negativos de entrenamiento no son rechazos explícitos.
- Las diferencias pequeñas no prueban superioridad estadística.

## Entrega 2: trabajo planificado

Dashboard Streamlit con catálogo, simulador de historial ordenado,
SASRec, líneas base, atención y métricas.

Analista con IA que recibe contexto estructurado y analiza las listas,
sin agregar, quitar ni reordenar recetas.

Innovación propuesta: auditor de popularidad, novedad y diversidad,
con indicadores calculados en Python y evaluación en 30 historiales.

Para historial vacío se mostrará popularidad como respaldo explícito.
La aplicación cargará el modelo en CPU y no entrenará durante su uso.

Esta fase todavía no está implementada.

## Uso de IA y participación

Se utilizó ChatGPT como apoyo para diseño, programación e interpretación.
Cada integrante debe comprender y revisar el trabajo presentado.

Los aportes de los integrantes deben documentarse mediante commits
reales realizados desde sus respectivas cuentas.

## Atribución

Los datos provienen de Food.com Recipes and Interactions.
Consultar la fuente para sus condiciones de uso y redistribución.
Este repositorio no concede derechos adicionales sobre los datos.