# Plan LOST - MountainCarContinuous

## 1. Alcance

LOST corresponde al ambiente `MountainCarContinuous-v0`. La entrega pide resolverlo con aprendizaje por refuerzo usando Q-Learning sobre observaciones y acciones continuas discretizadas, experimentar con hiperparametros y agregar Dyna-Q a partir de Sutton y Barto, capitulos 8.1 y 8.2.

La prioridad no es escribir mucho codigo, sino dejar un flujo reproducible para entrenar, comparar y documentar. El informe debe poder explicar que se probo, por que se probo, como se midio y que conclusion se obtuvo.

La fecha limite indicada en la letra es el 6 de julio de 2026 a las 21:00. Antes de eso debe quedar al menos un modelo computado guardado, porque la letra dice que si no se entrega un modelo para LOST, el ejercicio se considera no hecho.

## 2. Entregables de LOST

- Codigo Python completo para entrenar y evaluar Q-Learning.
- Codigo Python completo para entrenar y evaluar Dyna-Q.
- Notebook de ejecucion y analisis con graficos claros.
- Al menos un modelo guardado, por ejemplo `models/q_learning_best.pkl` o `models/dyna_q_best.pkl`.
- Resultados tabulados en `.csv` o `.json` para no depender solo de capturas.
- Graficos de recompensa, tasa de exito, pasos por episodio y comparacion de variantes.
- Seccion del informe con metodologia, parametros, tiempos de ejecucion, resultados, dificultades y uso de IA generativa.

## 3. Lectura tecnica minima

Material de clase relevante:

- Monte Carlo: aprendizaje a partir de episodios completos cuando no hay modelo del ambiente.
- TD(0): actualizacion paso a paso usando una estimacion del siguiente estado.
- Q-Learning: control TD off-policy con politica epsilon-greedy.
- Value Iteration: idea de maximizar el valor esperado usando Bellman; sirve como base conceptual, aunque en Mountain Car no tenemos el modelo exacto discretizado desde el inicio.

Formula base de Q-Learning:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

Idea base de Dyna-Q:

1. Ejecutar una accion real en el ambiente.
2. Actualizar Q con la transicion real.
3. Guardar la transicion en un modelo aprendido `modelo[(s,a)] = (r, s', done)`.
4. Repetir `n` veces: samplear una transicion guardada y hacer actualizaciones Q simuladas.

## 4. Entender el ambiente antes de entrenar

Primero ejecutar el notebook existente y registrar:

- Rango de observaciones: posicion y velocidad.
- Rango de acciones: fuerza continua entre valores negativos y positivos.
- Forma de la recompensa.
- Condicion de exito.
- Maximo de pasos por episodio.

Para Mountain Car Continuous, la estrategia aprendida debe descubrir que a veces conviene acelerar en sentido contrario para ganar impulso y luego subir. Esto es importante para interpretar graficos: al principio puede parecer que el agente "retrocede", pero eso puede ser correcto.

## 5. Estructura de codigo recomendada

Mantener la carpeta `MountainCarContinuous` simple y enfocada:

```text
MountainCarContinuous/
  q_learning_agent.py
  dyna_q_agent.py
  train.py
  evaluate.py
  experiment_configs.py
  continuous_mountain_car.ipynb
  models/
  results/
  plots/
```

Si no se quiere crear tantos archivos, se puede concentrar todo en `q_learning_agent.py` y el notebook, pero conviene separar entrenamiento, evaluacion y experimentos para documentar mejor.

## 6. Implementacion de discretizacion

Crear una clase o funciones puras para discretizar:

- `build_bins(position_bins, velocity_bins, action_bins)`
- `discretize_observation(obs) -> (pos_idx, vel_idx)`
- `action_index_to_value(action_idx) -> np.array([force])`
- `best_action(state) -> action_idx`
- `random_action() -> action_idx`

Puntos importantes:

- Usar `np.linspace` para crear cortes entre minimos y maximos del ambiente.
- Usar `np.digitize` o `np.searchsorted`.
- Hacer `clip` de indices para evitar salidas de rango.
- La tabla Q debe tener forma `(position_bins, velocity_bins, action_bins)`.
- Las acciones discretas deben cubrir valores negativos, cero y positivos.

Experimentos de discretizacion sugeridos:

| Variante | Posicion | Velocidad | Acciones | Motivo |
|---|---:|---:|---:|---|
| Baseline chico | 12 | 12 | 3 | Corre rapido y valida el flujo completo. |
| Intermedio | 20 | 20 | 5 | Balance razonable entre aprendizaje y detalle. |
| Fino | 30 | 30 | 7 | Mayor precision con mas costo de entrenamiento. |
| Muy fino | 40 | 40 | 9 | Solo si hay tiempo; puede aprender mas lento. |

No empezar con `100 x 100 x 10` como configuracion principal. Aunque aparece en el notebook, genera una tabla grande y muy dispersa para una primera version experimental.

## 7. Implementacion de Q-Learning

Completar `q_learning_agent.py` con:

- Constructor con parametros: bins, acciones, `alpha`, `gamma`, `epsilon_start`, `epsilon_min`, `epsilon_decay`, `seed`.
- Tabla Q inicializada en cero o con valores optimistas.
- Politica epsilon-greedy.
- Entrenamiento por episodios.
- Evaluacion sin exploracion, usando siempre `argmax`.
- Guardado y carga de modelo con `pickle`.
- Registro de metricas por episodio.

Flujo por episodio:

1. `obs, info = env.reset(seed=seed)`
2. Discretizar `obs`.
3. Elegir accion con epsilon-greedy.
4. Ejecutar `env.step(np.array([accion_continua]))`.
5. Discretizar siguiente observacion.
6. Actualizar Q.
7. Repetir hasta `terminated or truncated`.
8. Guardar recompensa total, cantidad de pasos, si llego al objetivo y epsilon usado.

Detalles que hay que cuidar:

- Gymnasium devuelve `terminated` y `truncated`; el episodio termina si cualquiera es verdadero.
- Para el bootstrap, si el episodio termino, usar target `reward`; si no, usar `reward + gamma * max(Q[next_state])`.
- Cuando haya empate en `argmax`, desempatar aleatoriamente entre las mejores acciones para evitar sesgos.
- Mantener seeds fijas por corrida para poder comparar.

## 8. Implementacion de Dyna-Q

Crear `dyna_q_agent.py` heredando o reutilizando la logica de Q-Learning.

Agregar:

- `planning_steps`: cantidad de actualizaciones simuladas por paso real.
- `model`: diccionario con clave `(state, action_idx)` y valor `(reward, next_state, done)`.
- Despues de cada paso real, guardar la transicion.
- Para cada paso de planificacion, samplear una clave conocida y aplicar la misma actualizacion Q.

Experimentos sugeridos:

| Variante | Planning steps | Objetivo |
|---|---:|---|
| Q-Learning puro | 0 | Baseline directo. |
| Dyna chico | 5 | Ver si mejora sin mucho costo. |
| Dyna medio | 10 | Candidato principal. |
| Dyna alto | 20 | Ver tradeoff entre tiempo y rendimiento. |
| Dyna muy alto | 50 | Solo si el tiempo lo permite. |

En el informe comparar Dyna-Q contra Q-Learning usando la misma discretizacion e hiperparametros base. La pregunta central es si Dyna-Q aprende mas rapido o con mejor rendimiento final por usar experiencias simuladas.

## 9. Hiperparametros a explorar

Primero validar que todo corre con pocos episodios. Luego hacer una grilla chica, no una busqueda enorme.

Parametros recomendados:

| Parametro | Valores iniciales |
|---|---|
| `alpha` | 0.05, 0.1, 0.2, 0.5 |
| `gamma` | 0.95, 0.99 |
| `epsilon_start` | 1.0 |
| `epsilon_min` | 0.01, 0.05 |
| `epsilon_decay` | 0.995, 0.999, 0.9995 |
| `episodes` | 2000, 5000, 10000 |
| `planning_steps` | 0, 5, 10, 20 |

Orden practico:

1. Fijar discretizacion intermedia `20 x 20 x 5`.
2. Probar 4 a 8 combinaciones de `alpha`, `gamma` y epsilon.
3. Elegir la mejor configuracion Q-Learning.
4. Con esa configuracion, comparar `planning_steps`.
5. Repetir solo las mejores variantes con 3 a 5 seeds distintas.

Evitar cambiar demasiadas cosas a la vez. Cada tabla o grafico debe tener una pregunta clara.

## 10. Metricas de evaluacion

Durante entrenamiento registrar:

- Recompensa total por episodio.
- Promedio movil de recompensa, por ejemplo ventana de 100 episodios.
- Cantidad de pasos por episodio.
- Exito o fracaso del episodio.
- Epsilon usado.
- Tiempo de entrenamiento.

Durante evaluacion registrar:

- Recompensa promedio en 20 o 50 episodios sin exploracion.
- Tasa de exito.
- Pasos promedio hasta terminar.
- Mejor, peor y desviacion estandar de recompensa.

Criterio de comparacion recomendado:

1. Tasa de exito alta.
2. Recompensa promedio alta.
3. Menos pasos promedio.
4. Menor tiempo de entrenamiento, si el rendimiento es comparable.

## 11. Graficos para el informe

Graficos minimos:

- Recompensa por episodio y promedio movil.
- Tasa de exito por bloques de episodios.
- Comparacion Q-Learning vs Dyna-Q.
- Comparacion de discretizaciones.
- Comparacion de `planning_steps`.

Graficos opcionales:

- Heatmap de `max_a Q(s,a)` sobre posicion y velocidad.
- Politica aprendida sobre el espacio discretizado, mostrando que accion elige en cada zona.
- Boxplot de recompensas de evaluacion por configuracion.

Cada grafico debe tener:

- Titulo claro.
- Ejes con unidades o significado.
- Leyenda si compara mas de una configuracion.
- Comentario de 2 a 4 lineas explicando que conclusion se saca.

## 12. Documentacion del informe

Estructura recomendada para la parte LOST:

1. Descripcion breve del ambiente.
2. Explicacion de por que se necesita discretizar.
3. Discretizacion elegida y alternativas probadas.
4. Implementacion de Q-Learning.
5. Experimentos de hiperparametros.
6. Implementacion de Dyna-Q.
7. Comparacion Q-Learning vs Dyna-Q.
8. Configuracion final elegida.
9. Dificultades encontradas.
10. Conclusiones.
11. Uso de IA generativa: indicar que Codex/ChatGPT se uso como apoyo para planificacion, estructura de codigo, debugging o redaccion, y que los resultados fueron verificados.

Preguntas que el informe debe responder:

- Como se transformo el problema continuo en discreto.
- Que impacto tuvo aumentar o reducir la discretizacion.
- Que hiperparametros fueron mas sensibles.
- Si Dyna-Q mejoro el aprendizaje y con que costo.
- Cual modelo se entrega finalmente y por que.

## 13. Orden de trabajo recomendado

### Paso 1 - Preparar base reproducible

- Crear carpetas `models`, `results` y `plots`.
- Definir seeds.
- Crear funciones de guardado de metricas.
- Dejar una corrida corta de 10 episodios para probar que no se rompe.

Resultado esperado: el entrenamiento corre de punta a punta aunque el agente todavia sea malo.

### Paso 2 - Q-Learning minimo

- Implementar discretizacion.
- Implementar epsilon-greedy.
- Implementar update de Q-Learning.
- Evaluar sin exploracion.
- Guardar un primer `.pkl`.

Resultado esperado: primer modelo entregable, aunque no sea el mejor.

### Paso 3 - Primeros experimentos

- Probar baseline chico e intermedio.
- Graficar recompensa y promedio movil.
- Confirmar que el agente mejora contra una politica aleatoria o contra su propio inicio.

Resultado esperado: primera evidencia para documentar.

### Paso 4 - Ajuste de hiperparametros

- Ejecutar grilla chica.
- Guardar resultados por configuracion.
- Elegir 2 o 3 candidatas.
- Repetir candidatas con varias seeds.

Resultado esperado: configuracion Q-Learning defendible.

### Paso 5 - Dyna-Q

- Implementar modelo aprendido.
- Probar `planning_steps = 5, 10, 20`.
- Comparar contra Q-Learning con la misma configuracion.

Resultado esperado: conclusion clara sobre si Dyna-Q conviene en este ambiente.

### Paso 6 - Modelo final y evidencia

- Entrenar configuracion final.
- Guardar modelo final.
- Ejecutar evaluacion sin exploracion.
- Exportar graficos finales.
- Registrar tiempo de ejecucion.

Resultado esperado: material listo para el informe.

### Paso 7 - Redaccion final

- Escribir metodologia.
- Insertar graficos y tablas.
- Explicar decisiones, no solo resultados.
- Agregar advertencias si alguna configuracion no funciono.
- Verificar que el zip incluya codigo, notebook, modelos e informe PDF.

## 14. Riesgos y como manejarlos

- El agente no aprende: reducir discretizacion, aumentar episodios, revisar epsilon decay y confirmar que las acciones tengan valores negativos y positivos.
- Aprende muy lento: usar Dyna-Q con `planning_steps` moderado o bajar cantidad de bins.
- Resultados muy variables: repetir con varias seeds y reportar promedio y desviacion.
- Modelo muy grande: guardar solo la tabla Q y la configuracion, no objetos de entorno.
- Graficos confusos: usar promedio movil y separar entrenamiento de evaluacion.
- Reward shaping tentador: si se usa, documentarlo como experimento separado. Para el resultado principal conviene usar la recompensa original del ambiente.

## 15. Checklist antes de entregar

- [ ] `q_learning_agent.py` implementado.
- [ ] `dyna_q_agent.py` implementado o Dyna-Q integrado de forma clara.
- [ ] Notebook ejecutable de punta a punta.
- [ ] Al menos un `.pkl` en `models/`.
- [ ] Resultados guardados en `results/`.
- [ ] Graficos guardados en `plots/`.
- [ ] Evaluacion final sin exploracion.
- [ ] Informe explica discretizacion, hiperparametros, Dyna-Q y resultados.
- [ ] Informe incluye uso de IA generativa.
- [ ] Zip final menor a 40 MB.

