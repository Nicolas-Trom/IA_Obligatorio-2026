# Plan de trabajo — Proyecto MATE (Martian Adversarial Tactics Engine)

> **Documento autónomo.** Contiene todo el contexto necesario para ejecutar el Proyecto MATE del Obligatorio de Inteligencia Artificial (Ing. en Sistemas, ORT). Está pensado para usarse como **contexto de una IA** que implemente las tareas, o como guía para el equipo. No depende de ningún otro documento.

---

## 0. Resumen ejecutivo

MATE consiste en implementar agentes inteligentes para el juego de tablero **Isolation**, donde el agente se enfrenta a un oponente en un escenario **adversarial**. El objetivo es tomar decisiones óptimas mediante algoritmos de búsqueda en árboles de juego.

El proyecto tiene **3 tareas obligatorias**:
1. **Técnicas:** implementar **Minimax** (con **Alpha-Beta Pruning**) y **Expectimax**, y decidir cuál conviene.
2. **Funciones de evaluación:** implementar y combinar heurísticas con distintos pesos.
3. **Experimentación:** definir pruebas para evaluar los agentes y registrar los resultados.

**Grupo de 2 integrantes** → no aplica la tarea extra de grupos de 3 (no se implementa MCTS).

**Meta medible:** que los agentes ganen consistentemente a `RandomAgent` y, sobre todo, al agente de referencia **`Stratagem`**.

---

## 1. Contexto del problema

La empresa ficticia *Red Destination™* anticipa que su rover en Marte deberá desenvolverse en "dinámicas lúdicas" frente a vida inteligente. Se usa un simulador del juego de mesa **Isolation**: el agente debe enfrentar a un oponente y **minimizar el riesgo de resultados desfavorables** mediante Minimax/Expectimax.

---

## 2. Ambiente y reglas

### El juego Isolation (según el código provisto)
- Tablero **4×4** (`Board`, por defecto `board_size=(4,4)`).
- Dos jugadores: **1 = `B`**, **2 = `R`**. Celdas eliminadas = **`X` (valor 3)**, celda vacía = `0`.
- Posiciones iniciales **aleatorias** (`place_players` mezcla posiciones).
- **Una jugada = `(dirección, celda_a_destruir)`**:
  - `dirección` ∈ `0..7`: `0` Up, `1` Down, `2` Left, `3` Right, `4` Up-Left, `5` Up-Right, `6` Down-Left, `7` Down-Right.
  - `celda_a_destruir` = tupla `(row, col)`.
  - Al jugar: el jugador **se mueve** una casilla en esa dirección (a una celda vacía), su **casilla previa queda eliminada** (`X`) y además **se destruye** la celda elegida (debe estar vacía).
- **Fin de partida:** pierde el jugador que **no puede moverse**. `is_end(player)` devuelve `(True, ganador)` donde el ganador es `player % 2 + 1` (el otro jugador).

### API reutilizable (NO reimplementar las reglas)
Archivo `Isolation/board.py` — clase `Board`:
- `get_possible_actions(player)` → lista de `(direccion, (row,col))` legales.
- `clone()` → copia del tablero. ⚠️ **instancia un `Board()` 4×4 por defecto** (no asumir otros tamaños sin ajustar).
- `play(action, player)` → aplica la jugada; devuelve `True`/`False` si fue válida.
- `is_end(player)` → `(done, ganador)`.
- `find_player_position(player)` → `(row, col)`.
- `render_grid()` → imprime el tablero en consola.
- `has_valid_moves(player)`, `can_move_to(...)`, `can_eliminate(...)` (auxiliares).

Archivo `Isolation/agent.py` — interfaz abstracta a heredar:
```python
class Agent(ABC):
    def __init__(self, player):
        self.player = player
        self.board = (player % 2) + 1     # id del rival
    def next_action(self, obs): ...        # obs ES un Board
    def heuristic_utility(self, board): ...
```

Archivo `Isolation/isolation_env.py` — wrapper Gym `IsolationEnv`:
- `reset()` → devuelve el `Board`.
- `step(action)` → `(grid, reward, done, winner, {})`.

Archivo `Isolation/play.py`:
- `play_vs_other_agent(env, agent1, agent2, render=False)` → juega una partida completa entre dos agentes.

Agentes provistos:
- `RandomAgent(player)` — elige una acción legal al azar.
- `InputAgent(player)` — juego manual por consola.
- **`Stratagem(player)`** — agente de **referencia a vencer**. Es un **Minimax ofuscado de profundidad 3** cuya heurística combina: **diferencia de movilidad** (libertades propias vs rival), **distancia al centro** y **distancia al oponente**.

### Cómo correr una partida
```python
from isolation_env import IsolationEnv
from random_agent import RandomAgent
from stratagem import Stratagem
from play import play_vs_other_agent

env = IsolationEnv()
play_vs_other_agent(env, agent1=RandomAgent(1), agent2=Stratagem(2), render=True)
```

### Setup (Poetry)
```bash
cd Isolation
poetry install
poetry run jupyter notebook   # isolation.ipynb
```

---

## 3. Tarea 1 — Minimax (con Alpha-Beta) y Expectimax

**Objetivo:** implementar `MinimaxAgent` y `ExpectimaxAgent` que hereden de `Agent`, y analizar cuál conviene para este juego. En Minimax es obligatorio usar **Alpha-Beta Pruning** y analizar su impacto.

### Diseño de clases
Ambos heredan de `Agent` e implementan:
- `next_action(obs)`: recibe el `Board` actual, ejecuta la búsqueda hasta una profundidad límite y devuelve la mejor acción `(direccion, (row,col))`.
- `heuristic_utility(board)`: función de evaluación del estado (ver Tarea 2).

### Generación de sucesores (reutilizar la API)
```python
for action in board.get_possible_actions(player):
    child = board.clone()
    child.play(action, player)     # aplica la jugada al clon
    # recursión sobre child con el otro jugador
```

### Minimax con Alpha-Beta (pseudocódigo)
```
function minimax(board, depth, alpha, beta, current_player):
    done, winner = board.is_end(current_player)
    if done:
        return (None, +∞ si gana YO else −∞)      # valor terminal
    if depth == 0:
        return (None, heuristic_utility(board))

    actions = board.get_possible_actions(current_player)
    best_action = None

    if current_player == self.player:             # nodo MAX
        value = −∞
        for a in actions:
            child = board.clone(); child.play(a, current_player)
            _, v = minimax(child, depth−1, alpha, beta, rival)
            if v > value: value, best_action = v, a
            alpha = max(alpha, value)
            if beta <= alpha: break               # poda β
        return best_action, value
    else:                                         # nodo MIN
        value = +∞
        for a in actions:
            child = board.clone(); child.play(a, current_player)
            _, v = minimax(child, depth−1, alpha, beta, self.player)
            if v < value: value, best_action = v, a
            beta = min(beta, value)
            if beta <= alpha: break               # poda α
        return best_action, value
```
`next_action` llama a `minimax(obs, max_depth, −∞, +∞, self.player)` y devuelve `best_action`.

> El agente `Stratagem` ya implementa internamente esta estructura (terminal → ±1/0, corte → `heuristic_utility`, recursión sobre `get_possible_actions` + `clone`/`play`). Sirve como referencia conceptual.

### Expectimax (pseudocódigo de la diferencia)
Igual que Minimax pero el **nodo del rival no minimiza, sino que promedia** (esperanza) sobre sus jugadas:
```
else:                                  # nodo CHANCE (rival)
    value = 0
    for a in actions:
        child = board.clone(); child.play(a, current_player)
        _, v = expectimax(child, depth−1, self.player)
        value += v
    return None, value / len(actions)  # esperanza (uniforme)
```
**Cuándo conviene Expectimax:** cuando el rival **no juega óptimo** (p. ej. `RandomAgent`), modelar sus jugadas como aleatorias da mejores decisiones que asumir el peor caso. Contra un rival fuerte/óptimo, Minimax (peor caso) es más seguro.

### Análisis del impacto de Alpha-Beta (obligatorio)
- A **igual profundidad**, comparar Minimax **con vs sin poda**:
  - **Nodos expandidos** (instrumentar un contador).
  - **Tiempo por jugada**.
- Mostrar el ahorro (Alpha-Beta puede reducir mucho el árbol sin cambiar el resultado).
- Discutir el efecto del **ordenamiento de jugadas** (*move ordering*): explorar primero las jugadas prometedoras mejora la poda.
- ⚠️ **Factor de ramificación:** `get_possible_actions` combina hasta 8 direcciones × celdas destruibles, por lo que el árbol crece rápido. Justificar la **profundidad máxima** elegida según el costo de tiempo por jugada.

---

## 4. Tarea 2 — Funciones de evaluación

**Objetivo:** implementar funciones de evaluación que valoren un estado, experimentar con combinaciones y ponderaciones.

### Catálogo de heurísticas a implementar
(alineadas con lo que hace `Stratagem`, más variantes)
| Heurística | Idea | Cálculo |
|---|---|---|
| Movilidad propia | tener más movimientos disponibles es bueno | `len(get_possible_actions(self.player))` |
| **Movilidad relativa** | ventaja de movilidad sobre el rival | `#mov_propios − #mov_rival` |
| Control del centro | el centro da más opciones | `−distancia(pos_propia, centro)` |
| Distancia al rival | acercarse/alejarse del rival | función de `distancia(pos_propia, pos_rival)` |
| Acorralar al rival | reducir libertades del rival | nº de celdas eliminadas/bloqueadas alrededor del rival |

### Diseño parametrizable
```python
heuristic_utility(board) = w1·h1(board) + w2·h2(board) + ... + wk·hk(board)
```
con pesos `w_i` configurables (p. ej. atributos del agente o un diccionario).

### Experimentación (a documentar)
- Probar heurísticas **aisladas** y en **combinación**, con distintos pesos.
- Tabla de **win-rate por configuración** de heurística/pesos.
- Discutir cómo cambia el **comportamiento** (agresivo = priorizar reducir movilidad del rival; posicional = priorizar centro/movilidad propia).

> ⚠️ En **estados terminales** la evaluación debe devolver un valor extremo coherente con la convención de signo (gana yo → `+∞`/valor muy alto; pierdo → `−∞`/valor muy bajo). No mezclar la escala de la heurística con la de los terminales sin cuidado.

---

## 5. Tarea 3 — Experimentación

**Objetivo:** definir pruebas para evaluar los agentes y hacer un **registro completo** de los resultados.

### Protocolo de torneo
Enfrentar cada agente/variante contra:
- `RandomAgent` (línea base baja).
- **`Stratagem`** (línea base fuerte, la referencia a vencer).
- **Entre variantes propias:** Minimax vs Expectimax, distintas heurísticas/pesos, distintas profundidades.

### Rigor estadístico
- **N partidas por enfrentamiento** (p. ej. 50–100): una sola partida no es concluyente porque las posiciones iniciales son **aleatorias**.
- **Alternar quién empieza** (empezar puede dar ventaja); idealmente jugar como jugador 1 la mitad de las partidas y como jugador 2 la otra mitad.
- Fijar/registrar **semillas** para reproducibilidad.

### Métricas a registrar
- **Win-rate** (% de victorias) por enfrentamiento.
- Profundidad usada.
- **Tiempo medio por jugada**.
- **Nodos expandidos** (y efecto de Alpha-Beta).

### Presentación
- **Tablas** de resultados (puede exportarse a CSV desde el notebook).
- **Gráficos:** barras de win-rate por agente/configuración, tiempo vs profundidad, nodos con/sin poda.
- Registro completo de los resultados, como exige la letra.

---

## 6. Entregables y checklist (MATE)

- [ ] `minimax_agent.py` (Minimax con Alpha-Beta) — u organización equivalente.
- [ ] `expectimax_agent.py` (Expectimax).
- [ ] Funciones de evaluación parametrizables (en los agentes o un módulo aparte).
- [ ] Notebook `.ipynb` con el torneo, las tablas y los gráficos.
- [ ] Secciones del informe correspondientes a MATE: técnica elegida y por qué, **impacto de Alpha-Beta** (nodos/tiempo), heurísticas y pesos probados, resultados de los torneos y notas de dificultades.

---

## 7. Notas para la IA ejecutora

- **Respetar la interfaz `Agent`:** `next_action(obs)` donde `obs` **es un `Board`**; `heuristic_utility(board)`. Heredar de `Agent` e implementar ambos métodos abstractos.
- **Reutilizar la lógica del juego** (`clone()`, `get_possible_actions()`, `play()`, `is_end()`, `find_player_position()`); **NO reimplementar las reglas**.
- ⚠️ `Board.clone()` crea un tablero **4×4 por defecto** — válido para el tamaño estándar; si se experimentara con otros tamaños habría que ajustar `clone()`.
- **No modificar** los archivos provistos `board.py` y `agent.py` (crear archivos nuevos para los agentes).
- Usar `play_vs_other_agent(env, a1, a2, render)` para las partidas; usar `render=False` para correr torneos rápido.
- Atención al **factor de ramificación**: limitar la profundidad para que el tiempo por jugada sea razonable; instrumentar contadores de nodos/tiempo para la Tarea 1.
- Cuidar la **convención de signo**: el valor terminal debe ser positivo cuando gana `self.player` y negativo cuando gana el rival, consistente en Minimax y Expectimax.

---

## 8. Recordatorios de la entrega (comunes al obligatorio)

- **Citar el uso de IA generativa:** indicar las herramientas usadas y el contexto (generación de ideas, redacción inicial, análisis, corrección de estilo, etc.). Todo contenido producido por IA debe ser revisado; los errores son responsabilidad del estudiante.
- **Formato de entrega:** un único archivo **`.zip` (≤ 40 MB)** con todo el código (`.py` y `.ipynb`) y el **informe en PDF de ≤ 20 páginas** (más anexos).
- **Fecha de entrega:** **06/07/2026 hasta las 21:00** vía gestion.ort.edu.uy. Defensa el 06/07/2026 (obligatoria y eliminatoria).
- **Cronograma sugerido:** empezar temprano — los torneos con muchas partidas y profundidad alta pueden tardar. Reservar tiempo para los experimentos de heurísticas y la comparación con/sin Alpha-Beta.
