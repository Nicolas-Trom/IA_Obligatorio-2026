# Plan de trabajo — Proyecto LOST (Learning-based Orientation and Steering for Traversal)

> **Documento autónomo.** Contiene todo el contexto necesario para ejecutar el Proyecto LOST del Obligatorio de Inteligencia Artificial (Ing. en Sistemas, ORT). Está pensado para usarse como **contexto de una IA** que implemente las tareas, o como guía para el equipo. No depende de ningún otro documento.

---

## 0. Resumen ejecutivo

LOST consiste en entrenar un **agente de aprendizaje por refuerzo tabular (Q-Learning)** que aprenda a controlar el ambiente **`MountainCarContinuous-v0`** de Gymnasium: un auto que debe subir una colina, pero cuyo motor no tiene fuerza suficiente para subir directamente, por lo que debe aprender a tomar impulso oscilando.

El proyecto tiene **4 tareas obligatorias**:
1. Discretizar observaciones y acciones (el ambiente es continuo).
2. Implementar **Q-Learning**.
3. Explorar **hiperparámetros** y justificar la elección.
4. Componente de investigación: implementar **Dyna-Q** (Sutton & Barto, cap. 8.1–8.2) y compararlo con Q-Learning.

**Grupo de 2 integrantes** → no aplica la tarea extra de grupos de 3.

> ⚠️ **Entregable obligatorio crítico:** se debe entregar **al menos un modelo computado (`.pkl`)** para este ejercicio. Si no se entrega, el ejercicio se considera **no hecho**.

---

## 1. Contexto del problema

La empresa ficticia *Red Destination™* desarrolla el rover "Out for Delivery" enviado a Marte. El rover aterrizó pero "no se encontró piloto": hay que construir un agente que aprenda a **navegar de forma autónoma**, descubriendo que **avanzar suele ser mejor que no hacerlo**.

Técnicamente esto se modela con `MountainCarContinuous-v0`.

### Por qué es difícil (entender esto es clave)
- La recompensa del ambiente es **dispersa y mayormente negativa**:
  - Cada paso resta `-0.1 · acción²` (penaliza gastar energía).
  - Solo al **alcanzar la meta** (posición ≥ 0.45) se otorga `+100`.
- Un agente aleatorio **casi nunca llega a la meta**, por lo que la señal de `+100` casi nunca aparece y la tabla Q apenas aprende.
- Consecuencia de diseño: hay que favorecer que el agente **alcance la meta al menos algunas veces** durante el entrenamiento. Estrategias: discretización adecuada, exploración prolongada (ε alto al inicio), muchos episodios y, opcionalmente, *reward shaping*.

---

## 2. Ambiente y setup

### Instalación (Poetry)
El ambiente ya viene preparado en la carpeta `MountainCarContinuous/` con su `pyproject.toml`:
- Python `~3.10`, `gymnasium ^1.0.0`, `numpy ^2.1`, `matplotlib ^3.9`, `pygame ^2.6`, `notebook`.

```bash
cd MountainCarContinuous
poetry install
poetry run jupyter notebook        # abre continuous_mountain_car.ipynb
# o para correr scripts:
poetry run python q_learning_agent.py
```

### API de Gymnasium (importante)
```python
import gymnasium as gym
import numpy as np

env = gym.make('MountainCarContinuous-v0', render_mode=None)  # None/'rgb_array' para entrenar rápido; 'human' para visualizar
obs, info = env.reset(seed=0)                  # obs = np.array([posicion, velocidad])
obs, reward, terminated, truncated, info = env.step(np.array([fuerza]))  # fuerza ∈ [-1, 1]
done = terminated or truncated
env.close()
```

> Nota: el notebook provisto usa la forma `obs, reward, done, _, _ = env.step(...)`. En Gymnasium el 3er valor es `terminated` y el 4º `truncated`; conviene manejar ambos (`done = terminated or truncated`) para no quedar en bucles infinitos cuando se alcanza el límite de pasos.

### Datos del ambiente
| Magnitud | Rango / valor |
|---|---|
| Posición | `[-1.2, 0.6]` |
| Velocidad | `[-0.07, 0.07]` |
| Acción (fuerza) | continua en `[-1, 1]` |
| Meta | posición `≥ 0.45` |
| Límite de pasos | ~999 por episodio (`truncated`) |
| Recompensa | `-0.1·acción²` por paso; `+100` al llegar a la meta |

### Archivos de trabajo
- `MountainCarContinuous/q_learning_agent.py`: clase `QLearningAgent` **vacía** (esqueleto) a completar. Firmas a respetar:
  ```python
  class QLearningAgent:
      def __init__(self): ...
      def next_action(self, obs): ...
      def train_agent(self, env, episodes=1000, epsilon=.9, gamma=.9, alpha=.99): ...
      def test_agent(self, env, episodes=10): ...
  ```
- `MountainCarContinuous/continuous_mountain_car.ipynb`: scaffolding ya provisto que conviene reutilizar:
  - `x_space = np.linspace(-1.2, 0.6, 100)`, `vel_space = np.linspace(-0.07, 0.07, 100)`
  - `get_state(obs)` con `np.digitize`
  - `actions = list(np.linspace(-1, 1, 10))`
  - `Q = np.zeros((len(x_space)+1, len(vel_space)+1, len(actions)))`
  - `optimal_policy(state, Q)` y `epsilon_greedy_policy(state, Q, epsilon)`
  - Un loop de episodio con el **update de Q por completar**.

---

## 3. Tarea 1 — Discretización de observaciones y acciones

**Objetivo:** convertir el espacio continuo (posición, velocidad) y la acción continua (fuerza) en conjuntos discretos, para poder usar una tabla Q. Explorar varias opciones y justificar.

### Sub-pasos
1. **Estados:** definir grillas con `np.linspace` para posición y velocidad y mapear una observación a un índice discreto con `np.digitize`:
   ```python
   x_space   = np.linspace(-1.2, 0.6, n_pos)
   vel_space = np.linspace(-0.07, 0.07, n_vel)

   def get_state(obs):
       x, vel = obs
       return np.digitize(x, x_space), np.digitize(vel, vel_space)
   ```
2. **Acciones:** discretizar `[-1, 1]` en `n_acciones` valores:
   ```python
   actions = list(np.linspace(-1, 1, n_acciones))
   real_action = np.array([actions[a_idx]])   # lo que recibe env.step
   ```
3. **Tabla Q:** dimensionar como `(n_pos+1, n_vel+1, n_acciones)`.

> ⚠️ **Detalle importante:** `np.digitize` con `len(space)` bordes devuelve índices en `0..len(space)` (incluye el "fuera de rango" por ambos extremos), por eso la tabla usa `+1` en cada dimensión de estado.

### Diseño experimental (a documentar)
Probar y comparar al menos **3 resoluciones de estado** y **3 tamaños de acción**:

| Config | n_pos × n_vel | n_acciones | Tamaño tabla Q | Observaciones |
|---|---|---|---|---|
| Gruesa | 20 × 20 | 5 | 21·21·5 = 2205 | aprende rápido, control burdo |
| Media | 40 × 40 | 10 | 41·41·10 ≈ 16 810 | equilibrio |
| Fina | 100 × 100 | 15 | 101·101·15 ≈ 153 015 | control fino, tarda en aprender |

Métricas a comparar por configuración: tamaño/memoria de la tabla, velocidad de convergencia (episodios hasta resolver), recompensa final, % de éxito.

### Justificación a redactar en el informe
- **Trade-off granularidad ↔ tamaño del espacio de estados ↔ datos necesarios** (maldición de la dimensionalidad): más bins = política más precisa pero muchísimos más estados que visitar para aprender.
- Discutir **bins uniformes vs no uniformes** (p. ej. más resolución en velocidades cercanas a 0 o cerca de la meta, donde la decisión es más sensible).
- Justificar la configuración final elegida con datos.

---

## 4. Tarea 2 — Q-Learning

**Objetivo:** completar `QLearningAgent` con Q-Learning tabular funcional y guardar el modelo entrenado.

### Estructura propuesta
- Atributos en `__init__`: las grillas (`x_space`, `vel_space`), la lista `actions`, y la tabla `Q`.
- `next_action(obs)`: política **greedy** (para test/uso): `actions[np.argmax(Q[get_state(obs)])]`.
- `train_agent(...)`: bucle de entrenamiento con política ε-greedy.
- `test_agent(...)`: corre N episodios con política greedy y reporta recompensa media / % éxito.

### Algoritmo (pseudocódigo)
```
inicializar Q en ceros (o pequeños valores)
para cada episodio en 1..episodes:
    obs, _ = env.reset()
    s = get_state(obs)
    done = False
    while not done:
        # ε-greedy
        if random() < epsilon: a_idx = random index
        else:                  a_idx = argmax(Q[s])
        obs2, r, terminated, truncated, _ = env.step([actions[a_idx]])
        done = terminated or truncated
        s2 = get_state(obs2)
        # update Q-Learning
        target = r if terminated else r + gamma * max(Q[s2])
        Q[s][a_idx] += alpha * (target - Q[s][a_idx])
        s = s2
        acumular reward
    # decaer epsilon (y opcionalmente alpha)
    epsilon = max(eps_min, epsilon * decay)
    registrar reward del episodio
```

### Regla de actualización
```
Q[s, a] ← Q[s, a] + α · ( r + γ · maxₐ' Q[s', a'] − Q[s, a] )
```
En estado **terminal** (llegó a la meta), el target es solo `r` (no se suma el futuro).

### Detalles y buenas prácticas
- **ε-greedy con decaimiento:** empezar con ε alto (≈0.9–1.0) para explorar y decaer hacia ε_min (≈0.01–0.05). Sin exploración suficiente el agente nunca descubre la meta.
- **γ cercano a 1** (p. ej. 0.95–0.99): el horizonte es largo y la recompensa llega al final.
- **α (learning rate):** valores moderados (0.1–0.5); el `0.99` del esqueleto es alto, conviene revisarlo. Opcionalmente decaer α.
- **Reward shaping (opcional, discutir pros y contras):** para densificar la señal se puede sumar una recompensa basada en energía mecánica / altura / `|velocidad|`. ⚠️ Riesgo: cambiar la política óptima si el shaping no es *potential-based* (`F(s') - F(s)`). Documentar si se usa o no y por qué.
- **Reproducibilidad:** fijar `np.random.seed(...)`, `random.seed(...)` y `env.reset(seed=...)`.

### Persistencia del modelo (OBLIGATORIO)
```python
import pickle
with open('q_table.pkl', 'wb') as f:
    pickle.dump({'Q': Q, 'x_space': x_space, 'vel_space': vel_space, 'actions': actions}, f)
```
Guardar junto con la tabla **la discretización** usada (grillas y acciones) para poder reconstruir el agente. `test_agent` debe poder cargar este `.pkl`.

### Qué documentar
- Configuración final, curva de recompensa por episodio, recompensa de test, % de éxito, tiempo de entrenamiento.

---

## 5. Tarea 3 — Exploración de hiperparámetros

**Objetivo:** experimentar con múltiples combinaciones de hiperparámetros, justificar la métrica de evaluación y la elección final.

### Espacio de búsqueda
| Hiperparámetro | Valores sugeridos |
|---|---|
| `alpha` (learning rate) | 0.1, 0.2, 0.5 |
| `gamma` (descuento) | 0.95, 0.99 |
| `epsilon` inicial | 0.9, 1.0 |
| `epsilon` decay / `eps_min` | 0.999 / 0.01, 0.995 / 0.05 |
| nº de episodios | 2 000, 5 000, 10 000 |
| discretización | atada a la Tarea 1 (gruesa / media / fina) |

### Estrategia
- **Grid search acotado** o búsqueda manual guiada (no hace falta combinatoria completa).
- Fijar semilla y promediar **varias corridas** por configuración para reducir varianza.
- Reportar **tiempo de ejecución** por configuración.

### Métrica de evaluación (justificar)
Evaluar con **política greedy (ε=0)** durante N episodios de test:
- **Recompensa promedio de test** (métrica principal).
- **% de éxito** (proporción de episodios que alcanzan la meta).
- **Pasos medios hasta la meta** (eficiencia).

Documentar **por qué** se elige esa métrica (la recompensa de entrenamiento está contaminada por la exploración; la de test refleja la política aprendida).

### Visualizaciones requeridas
- **Curvas de aprendizaje:** recompensa por episodio + media móvil.
- **Heatmap de la política / función de valor** sobre el espacio (posición × velocidad): muestra qué acción/valor aprendió el agente en cada región.
- **Tabla resumen** comparando configuraciones.

---

## 6. Tarea 4 — Componente de investigación: Dyna-Q

**Objetivo:** leer Sutton & Barto, *Reinforcement Learning: An Introduction*, cap. **8.1 y 8.2**, implementar **Dyna-Q** y compararlo con Q-Learning de forma análoga.

### Concepto (resumen para el informe)
Dyna-Q integra tres procesos sobre la misma tabla Q:
1. **Aprendizaje directo (RL):** update de Q-Learning con la experiencia real.
2. **Modelo del ambiente:** se aprende `Model[s, a] → (r, s')` a partir de la experiencia.
3. **Planning:** tras cada paso real, se hacen `n` updates "simulados" muestreando pares `(s, a)` ya visitados y usando el modelo. Esto reutiliza la experiencia → mayor **sample-efficiency** (aprende con menos pasos reales).

### Algoritmo Dyna-Q (pseudocódigo)
```
inicializar Q y Model (dict)
para cada episodio:
    s = estado inicial
    while not done:
        a = ε-greedy(Q, s)
        r, s', done = step real
        # (1) aprendizaje directo
        Q[s,a] += α (r + γ max Q[s'] − Q[s,a])
        # (2) actualizar modelo
        Model[(s,a)] = (r, s')
        # (3) planning: n updates simulados
        repetir n veces:
            (sp, ap) = muestrear par visitado al azar de Model
            (rp, sp2) = Model[(sp, ap)]
            Q[sp,ap] += α (rp + γ max Q[sp2] − Q[sp,ap])
        s = s'
```

### Diseño experimental
- Reutilizar **la misma discretización** y **la misma métrica** de la Tarea 3 para una comparación justa.
- Comparar **Dyna-Q vs Q-Learning con el mismo presupuesto de pasos/episodios reales**.
- Variar el nº de pasos de planning `n` ∈ {0, 5, 20, 50} (con `n=0` Dyna-Q ≡ Q-Learning).
- **Gráfico comparativo:** curvas de aprendizaje (recompensa / pasos para resolver vs episodios) mostrando que mayor `n` acelera la convergencia en pasos reales.
- Discutir el **costo computacional extra** del planning frente a la ganancia en sample-efficiency.

> Nota: en un ambiente determinista como MountainCar el modelo tabular es exacto, lo que hace a Dyna-Q especialmente efectivo. Mencionarlo.

---

## 7. Entregables y checklist (LOST)

- [ ] `q_learning_agent.py` completo (respetando las firmas del esqueleto).
- [ ] Agente **Dyna-Q** (`.py` o sección de notebook).
- [ ] Notebook(s) `.ipynb` con los experimentos de las 4 tareas y los gráficos.
- [ ] **Al menos un modelo `.pkl`** entrenado (OBLIGATORIO).
- [ ] Secciones del informe correspondientes a LOST: resumen del abordaje por tarea, parámetros usados, tiempos de ejecución, resultados, gráficos y notas de dificultades.

---

## 8. Notas para la IA ejecutora

- **Respetar las firmas** del esqueleto `QLearningAgent` (`__init__`, `next_action(obs)`, `train_agent(env, episodes, epsilon, gamma, alpha)`, `test_agent(env, episodes)`); no cambiar la API pública.
- Trabajar con rutas relativas a `MountainCarContinuous/`.
- **Entrenar con `render_mode=None`** (mucho más rápido); usar `'human'` solo para demos puntuales.
- Manejar `terminated`/`truncated` por separado: el target del update solo descarta el futuro cuando `terminated` (llegó a meta), no cuando `truncated` (se acabó el tiempo).
- Fijar semillas (`np.random.seed`, `random.seed`, `env.reset(seed=...)`) para reproducibilidad.
- No romper la compatibilidad con `continuous_mountain_car.ipynb` provisto (reutilizar sus utilidades).
- Guardar en el `.pkl` también la discretización (grillas + acciones), no solo la matriz Q.

---

## 9. Recordatorios de la entrega (comunes al obligatorio)

- **Citar el uso de IA generativa:** indicar las herramientas usadas y el contexto (generación de ideas, redacción inicial, análisis, corrección de estilo, etc.). Todo contenido producido por IA debe ser revisado; los errores son responsabilidad del estudiante.
- **Formato de entrega:** un único archivo **`.zip` (≤ 40 MB)** con todo el código (`.py` y `.ipynb`), los modelos (`.pkl`) y el **informe en PDF de ≤ 20 páginas** (más anexos).
- **Fecha de entrega:** **06/07/2026 hasta las 21:00** vía gestion.ort.edu.uy. Defensa el 06/07/2026 (obligatoria y eliminatoria).
- **Cronograma sugerido:** empezar temprano — los entrenamientos (especialmente con discretización fina y muchos episodios) pueden tardar bastante. Reservar tiempo para correr los experimentos de hiperparámetros y Dyna-Q.
