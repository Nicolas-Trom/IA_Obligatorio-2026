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
