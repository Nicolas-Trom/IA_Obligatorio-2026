# Bitácora de decisiones — LOST (MountainCarContinuous)

> Registro incremental de situaciones que surgieron, alternativas consideradas y
> la decisión tomada (con su justificación). El código conserva solo la solución
> final; este documento conserva el **camino**. Es materia prima directa para el
> informe (la letra pide explícitamente documentar parámetros, tiempos,
> resultados y dificultades encontradas).

---

## 0. Hechos verificados del ambiente

Confirmados corriendo el simulador (`MountainCarContinuous-v0`, gymnasium 1.3.0):

- **Observación:** `[posición, velocidad]`, posición ∈ [-1.2, 0.6], velocidad ∈ [-0.07, 0.07].
- **Acción:** fuerza continua ∈ [-1.0, 1.0].
- **Recompensa:** `-0.1 · acción²` por paso **+100** al alcanzar la meta (posición ≥ 0.45).
- **Sin penalización por tiempo.** (clave, ver Decisión 3).
- **Máx. pasos por episodio:** 999.

---

## Decisión 1 — Estructura de código modular

**Situación:** el material de cátedra entrega un `q_learning_agent.py` con la clase
vacía y un notebook con funciones sueltas.

**Alternativas:**
- (A) Concentrar todo en `q_learning_agent.py` + notebook (menos archivos).
- (B) Modular: agente, Dyna-Q, discretización, configs, train, evaluate separados.

**Decisión:** (B) modular. **Por qué:** facilita documentar los experimentos por
separado, comparar variantes y que Dyna-Q herede de Q-Learning sin duplicar lógica.

---

## Decisión 2 — API del agente rediseñada

**Situación:** el esqueleto de cátedra tiene `train_agent(env, episodes, epsilon,
gamma, alpha)` con ε **fijo** y sin lugar natural para la discretización, las
métricas ni el guardado.

**Decisión:** rediseñar con los hiperparámetros en el **constructor** y métodos
`train` / `evaluate` / `save` / `load`. **Por qué:** el plan requiere ε con
*decay*, métricas por episodio, guardado `.pkl` y Dyna-Q; todo eso encaja mejor con
un agente configurado al construirse. El notebook de cátedra no referencia esos
métodos, así que no se rompe nada.

---

## Decisión 3 — El agente aprende a "no hacer nada" (óptimo local)

### Situación observada

Baseline Q-Learning con discretización intermedia (`20×20×5`, α=0.1, γ=0.99,
ε: 1.0→0.05, 3000 episodios, 61 s):

| Episodios | reward medio (100) | tasa de éxito | ε |
|---:|---:|---:|---:|
| 300  | -38.97 | 0.00 | 0.74 |
| 1500 | -11.87 | 0.00 | 0.22 |
| 3000 | **-2.67** | **0.00** | 0.05 |

- **Evaluación greedy (20 ep):** reward 0.00, 999 pasos, **0 % de éxito**.
- El agente **nunca alcanza la meta**. La "mejora" de la recompensa (de -39 a -2.67)
  no es aprendizaje de la tarea: es convergencia al óptimo local de **quedarse quieto**
  (fuerza 0 → penalización `-0.1·0² = 0`).

### Causa (diagnóstico)

El ambiente **no penaliza el tiempo**. Como "no moverse" cuesta 0, es óptimo
mientras el `+100` de la meta no aparezca. Con exploración ε-greedy de acciones
discretas `{-1, 0, 1}` independientes, construir el impulso coordinado (~100+ pasos)
para subir es extremadamente improbable → el `+100` casi nunca se observa → la tabla
Q nunca lo propaga.

**Contraste con Cliff Walking (Práctico 6):** ese ambiente **sí** aprende con
Q-Learning porque da **-1 por paso**. Esa penalización densa empuja a terminar rápido
y premia todo progreso hacia la meta. MountainCarContinuous carece de esa señal densa.
El material de clase (teórico de Q-Learning y práctico de Cliff Walking) no contiene
una solución directa, pero el contraste señala qué falta: **reintroducir una señal
densa que incentive el progreso.**

### Alternativas consideradas

1. **Reward shaping basado en energía (potential-based).** Sumar durante el
   entrenamiento `F = γ·Φ(s') − Φ(s)` con `Φ` proporcional a la energía mecánica
   (altura + velocidad²). Premia acumular impulso antes de llegar a la meta.
   *Ventaja:* teóricamente **no cambia la política óptima** (Ng, Harada & Russell,
   1999). Resuelve incentivo y descubrimiento.
2. **Shaping heurístico simple** (`reward += k·|velocidad|`). Fácil y efectivo, pero
   técnicamente altera la política óptima.
3. **Penalización por tiempo** (`reward −= c` por paso, análogo a Cliff Walking).
   Hace costoso "no hacer nada", pero por sí sola no resuelve el *descubrimiento*
   de la meta.
4. **Mejor exploración con reward original** (optimistic init, repetición de acción,
   ε alto sostenido). Mantiene el reward original pero es menos confiable.

### Decisión

**Opción 1 — reward shaping basado en energía (potential-based).** Se mantiene además
el **baseline con reward original** como comparación (documenta el fracaso, que es
evidencia válida). **Por qué:** es la más robusta —densifica la señal *y* da gradiente
para construir impulso— y la mejor de narrar: reintroduce densidad de señal como el
-1/paso de Cliff Walking, sin alterar la política óptima.

Potencial usado: `Φ(x,v) = escala · (sin(3x) + (v/0.07)²)` (energía mecánica:
altura de la colina + energía cinética normalizada). Ver `reward_shaping.py`.

### Resultado

Con la discretización intermedia (`20×20×5`, α=0.1, γ=0.99, 2000 episodios) el
shaping resuelve el problema. **Las métricas usan el reward original** (el shaping
solo guía el aprendizaje) y la **evaluación es sobre el ambiente sin shaping**:

| Escala | tasa de éxito | reward eval | pasos eval | std |
|---:|---:|---:|---:|---:|
| 1  | 0.00 | 0.00  | 999 | — |
| 5  | 1.00 | 92.00 | 156 | ±0.26 |
| 10 | 1.00 | 92.35 | 109 | ±1.11 |

- Con **escala 1 no aprende**: la señal densa es demasiado débil para vencer el
  atractor de "no hacer nada". Hace falta que el shaping pese lo suficiente (≥5).
- Con **escala ≥5** aprende al 100 %: pasa de 999 pasos (timeout) a ~110-156 pasos
  y de reward 0 a ~92 (= +100 de la meta menos el costo de energía).
- Escala 10 llega más rápido (109 pasos) pero con más varianza; escala 5 es más
  estable. **Elección por defecto: escala 5** (se puede revisar en la comparación
  final con más semillas).
- Tiempo de entrenamiento: ~18-22 s por corrida de 2000 episodios.

**Aprendizaje transferido de Cliff Walking:** el rol del shaping por energía aquí es
análogo al del -1/paso allá — dar señal densa de progreso.

---

## Decisión 4 — Dyna-Q vs Q-Learning (componente de investigación)

**Objetivo:** ver si Dyna-Q (Sutton & Barto, cap. 8) aprende más rápido que
Q-Learning por reutilizar experiencia simulada. Dyna-Q guarda cada transición real
en un modelo `(s,a) → (r, s', terminal)` y, tras cada paso real, hace
`planning_steps` actualizaciones extra sampleando transiciones ya vistas.

**Montaje:** misma discretización (`20×20×5`), mismos hiperparámetros y shaping
(escala 5), 2000 episodios. Único cambio: `planning_steps` = 0 (Q-Learning), 5, 10, 20.
Gráfico: `plots/qlearning_vs_dynaq.png`.

**Resultados:**

| Variante | Episodios ≈ para 100% éxito | Tiempo entren. | reward eval | pasos eval |
|---|---:|---:|---:|---:|
| Q-Learning (0) | ~900 | 35 s | 92.00 | 156 |
| Dyna-Q 5 | ~450 | 53 s | 93.54 | 100 |
| Dyna-Q 10 | ~500 | 154 s | 91.03 | 141 |
| Dyna-Q 20 | ~650 | 142 s | 91.78 | 152 |

**Conclusiones:**
- **Dyna-Q aprende en menos episodios** que Q-Learning puro (converge ~2× antes).
  Confirma el beneficio del capítulo 8: la experiencia simulada acelera el aprendizaje.
- **Tradeoff:** ese beneficio se paga en **tiempo de máquina** (planning 10/20 tardan
  ~4× más en segundos). Como el simulador es barato, en *tiempo de reloj* Q-Learning
  puro conviene aquí; Dyna-Q rinde cuando la experiencia real es cara.
- **Más planning no fue mejor:** Dyna-Q 5 fue el más rápido en converger, por encima
  de 10 y 20. Resultado de **una sola semilla**: hay que confirmarlo repitiendo con
  varias semillas antes de afirmarlo en el informe.
- Todas alcanzan 100 % de éxito y reward ~92 en evaluación: el rendimiento final es
  equivalente; la diferencia está en la **velocidad de aprendizaje**, no en la calidad.

*(Pendiente: repetir con 3-5 semillas para robustez.)*

---

## Decisión 5 — Impacto de la discretización (tarea 1)

**Objetivo:** justificar la elección de la grilla midiendo cómo la resolución
afecta el aprendizaje. Todo igual (shaping escala 5, α=0.1, γ=0.99, 3000 episodios)
salvo la cantidad de casilleros. Gráfico: `plots/comparacion_discretizaciones.png`.

| Grilla (pos×vel×acc) | Casilleros | Éxito | reward eval | pasos eval | Tiempo |
|---|---:|---:|---:|---:|---:|
| Chico 12×12×3 | 432 | **100 %** | 93.33 | 72 | 46 s |
| Intermedio 20×20×5 | 2.000 | **100 %** | 92.90 | 154 | 41 s |
| Fino 30×30×7 | 6.300 | 0 % | -0.02 | 999 | 104 s |
| Muy fino 40×40×9 | 14.400 | 0 % | 0.00 | 999 | 107 s |

**Conclusiones:**
- **La resolución fina no aprendió** en 3000 episodios: cuantos más casilleros, más
  dispersa la tabla y menos visitas por casillero → no llega a llenarse. El "fino"
  muestra ráfagas de aprendizaje que no sostiene; el "muy fino" queda plano en 0 %.
- **Las grillas chicas aprenden bien y rápido.** La más chica (12×12×3) fue la más
  veloz en converger y la que menos pasos usa para llegar (72). Para este problema,
  el "punto justo" está del lado **grueso**.
- **Matiz honesto para el informe:** las grillas finas probablemente *sí* aprenderían
  con muchos más episodios; el hallazgo es "bajo un presupuesto fijo de episodios,
  más resolución = peor", no "más resolución es imposible".
- **Elección:** se toma la **intermedia (20×20×5)** como configuración base — buen
  balance entre resolución y velocidad de aprendizaje — dejando registrado que la
  chica también rinde muy bien.

---

## Decisión 6 — Grilla de hiperparámetros (tarea 3)

**Objetivo:** justificar la elección de α (tasa de aprendizaje), γ (descuento) y el
decay de ε (exploración). Grilla chica: se parte de la base (α=0.1, γ=0.99,
decay=0.999) y se cambia **una perilla por vez**. Todo lo demás igual (20×20×5,
shaping escala 5, 2000 episodios). Gráfico: `plots/comparacion_hiperparametros.png`.

| Config | Cambio | Éxito | reward eval | pasos |
|---|---|---:|---:|---:|
| α=0.05 | α baja | **0 %** | 0.00 | 999 |
| α=0.10 (base) | — | 100 % | 92.00 | 156 |
| α=0.20 | α alta | 100 % | 91.11 | 150 |
| γ=0.95 | γ baja | 100 % (inestable) | 93.32 | 188 |
| decay lento (0.9995) | ε explora más | 100 % | 90.46 | 158 |

**Conclusiones:**
- **α es el hiperparámetro más sensible.** Con α=0.05 el paso de actualización es
  tan chico que **no llega a aprender** en 2000 episodios. Con α=0.10–0.20 converge
  bien; α=0.20 fue el más rápido en este montaje.
- **γ=0.99 es importante para la estabilidad.** Con γ=0.95 la curva oscila mucho
  (sube y baja): al estar el premio (+100) lejos, un descuento más agresivo debilita
  la propagación del valor y vuelve la política temblorosa.
- **Explorar de más retrasa:** con decay más lento (ε alto por más tiempo) la
  convergencia es más lenta, sin mejor resultado final.
- **Elección final: α=0.10, γ=0.99, decay=0.999** (equilibrio y estabilidad).
  α=0.20 queda como alternativa algo más rápida.

*(Pendiente: confirmar con varias semillas antes de cerrar la elección en el informe.)*

---

## Decisión 7 — ¿Una grilla más grande mejoraría con más episodios?

**Pregunta:** las grillas finas fallaron con 3000 episodios (Decisión 5). ¿Aprenderían
—y superarían a la chica— si les damos muchos más episodios? Se probó la más grande,
`40×40×9` (14.400 casilleros), con **8000 episodios** (2.7× el presupuesto anterior),
mismos hiperparámetros y shaping. Gráfico: `plots/grilla_grande_8k.png`.

**Resultado:** **siguió dando 0 % de éxito.** El reward volvió a converger a ~0 →
el agente cayó de nuevo en "no hacer nada". Tiempo: 222 s.

**Diagnóstico:** no fue solo falta de datos. Con `epsilon_decay=0.999`, ε llega a su
mínimo (0.01) cerca del episodio 4600. En una tabla tan gigante, esa ventana de
exploración **no alcanzó** para hallar un camino confiable; una vez que ε se congela,
el agente deja de explorar y queda atrapado en el óptimo local. Los ~3400 episodios
restantes ya no exploran → no aportan.

**Conclusiones (fuertes para el informe):**
- **Más episodios por sí solos no arreglan una grilla muy fina.** Habría que re-tunear
  *toda la receta* (ε que baje más lento, y probablemente muchos más episodios todavía).
  Es un pozo de esfuerzo para, en el mejor caso, **igualar** a la grilla chica —que ya
  está cerca del techo de recompensa (~92 de ~100).
- **La grilla chica "funciona de fábrica"** con los hiperparámetros por defecto; la
  fina no. Eso justifica preferir la resolución baja/intermedia.
- **Responde el "¿por qué no una grilla más grande?"** con evidencia: probamos, y no
  solo no mejora, sino que ni siquiera aprende sin retrabajar la configuración. (Esto
  también explica por qué un enfoque tipo 100×100×15 *necesita* ~10.000 episodios y
  ajustes: no es que sea mejor, es que su propia elección de grilla se lo exige.)
### Segundo intento de rescate: ε que baja más lento

Para responder "¿y si el problema era solo la exploración?", se reentrenó la misma
`40×40×9` con **ε de decaimiento lento (0.9995, ~2× más exploración)** y **12.000
episodios**. Gráfico: `plots/grilla_grande_rescate.png`.

**Resultado: siguió en 0 % de éxito.** Hubo destellos de éxito durante la exploración
temprana (ep. 1200-3600, ~1-2 %), pero **nunca se afirmó una política**; al bajar ε,
volvió a "no hacer nada" (reward ~0). Tiempo: 256 s.

**Conclusión reforzada:** probamos **dos** arreglos razonables (más episodios; y más
exploración + más episodios) y **los dos fallaron**. La grilla muy fina no es "un poco
más cara": es **sustancialmente más difícil de hacer andar**, requiere retrabajar la
receta a fondo (y quizás decenas de miles de episodios), para —en el mejor caso—
**igualar** a la grilla chica, que ya está cerca del techo. La chica "funciona de
fábrica"; la fina es un pozo de esfuerzo sin premio. **Matiz honesto:** no probamos
que sea imposible; con ajustes mucho más agresivos podría llegar, pero no vale la pena.

---

## Decisión 8 — Robustez con varias semillas

**Objetivo:** todas las conclusiones anteriores salían de **una sola semilla** (una
"tirada de dado"). Para ver si eran robustas o casualidad, se repitieron las 4 configs
de Q-Learning vs Dyna-Q con **5 semillas** cada una (código: `seeds_experiment.py`,
20 corridas, ~30 min). Gráfico: `plots/comparacion_semillas.png` (media entre semillas
± desvío).

**Resultado de evaluación (media ± desvío entre 5 semillas):**

| Config | reward eval | éxito eval | semillas fallidas |
|---|---:|---:|---|
| Q-Learning (0) | 69.01 ± 46.40 | 0.80 ± 0.40 | **1 de 5** (0% éxito, reward -24) |
| Dyna-Q 5 | 92.75 ± 0.86 | **1.00 ± 0.00** | ninguna |
| Dyna-Q 10 | 89.58 ± 3.16 | 0.99 ± 0.02 | ninguna |
| Dyna-Q 20 | 80.05 ± 21.88 | 0.94 ± 0.12 | 1 de 5 (70% éxito) |

**Conclusiones (importantes, y que una sola semilla ocultaba):**
- **Q-Learning puro es frágil:** en 1 de 5 semillas quedó con una política rota (0% de
  éxito en evaluación). Con una sola corrida "afortunada" parecía tan bueno como Dyna-Q;
  con 5 semillas se ve que **su varianza es enorme** (éxito 0.80 ± 0.40).
- **Dyna-Q 5 es el más robusto:** 5/5 semillas al 100%, desvío mínimo (±0.86). El
  beneficio de Dyna-Q **no es solo velocidad, sino confiabilidad**: la experiencia
  simulada estabiliza el aprendizaje.
- **Más planning no es más estable:** Dyna-Q 20 también tuvo una semilla mala. El punto
  justo sigue siendo **planning ≈ 5** (ahora confirmado entre semillas, no por suerte).
- **Valor de este experimento:** justifica por qué reportar **media ± desvío** y no una
  sola corrida — cambia la lectura de "todos llegan al 100%" a "Q-Learning puede fallar;
  Dyna-Q 5 es confiable".

**Impacto en el modelo final:** el modelo entregado `shaped_s5` (Q-Learning, semilla 0)
funciona (100%), pero esta evidencia sugiere que **Dyna-Q con planning=5 es la receta de
entrenamiento más confiable**. Se deja documentado para la elección final del informe.
