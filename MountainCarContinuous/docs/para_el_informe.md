# Guía para redactar el informe — LOST

> Documento de arranque para escribir la sección LOST del informe. Empezá leyendo
> `docs/bitacora.md`: sus **9 decisiones** (situación → alternativas → decisión →
> resultados) son el esqueleto del informe.

## Alcance y formato

- Escribir **solo la parte LOST**. La parte **MATE** la hace el compañero de grupo.
- Entre LOST + MATE: informe **≤ 20 páginas** + anexos, en **PDF**, dentro de un `.zip` ≤ 40 MB.
- Requisitos (de `Obligatorio 2026 Marzo.pdf`): resumen de cómo se abordó cada tarea
  (interacción con el simulador, **parámetros, tiempos de ejecución, resultados**),
  **gráficos claros con comentarios**, **notas de dificultades**, y **citar el uso de
  IA generativa** (se usó como apoyo para planificar, codear, depurar y redactar; todo
  verificado por el grupo).

## Artefactos (¡OJO: están gitignorados, no aparecen en git!)

Existen localmente en la carpeta `MountainCarContinuous/`. Hay que mirarlos ahí:

### Gráficos (`plots/`) — son las figuras del informe
- `curva_shaped_s5.png` — curva de aprendizaje del modelo final (éxito 0→100%).
- `comparacion_discretizaciones.png` — impacto de la grilla (chicas aprenden, finas no).
- `qlearning_vs_dynaq.png` — Q-Learning vs Dyna-Q (Dyna-Q aprende más rápido en episodios).
- `comparacion_hiperparametros.png` — grilla de hiperparámetros (α sensible, γ=0.99 estable).
- `comparacion_semillas.png` — robustez con bandas (Q-Learning frágil, Dyna-Q 5 confiable).
- `grilla_grande_8k.png` y `grilla_grande_rescate.png` — la grilla 40×40×9 no mejora ni
  con más episodios ni con más exploración.

### Modelos (`models/*.pkl`) y resultados (`results/*.json`)
- Modelo final a entregar: `shaped_s5.pkl` (Q-Learning) + `plan05_dyna.pkl` (Dyna-Q 5).
- Cada `results/<name>.json` tiene config + métricas por episodio + evaluación.

## Resultados clave (headline)

- **Ambiente:** `MountainCarContinuous-v0`. Reward `-0.1·acción²` por paso, +100 en meta,
  sin penalización por tiempo → con reward original el agente cae en "no hacer nada".
- **Solución:** reward shaping por energía (potential-based, escala 5) → 0% a 100% de éxito.
- **Discretización elegida:** `20×20×5`. Las finas (30×30×7, 40×40×9) no aprenden con
  presupuesto razonable (Decisiones 5 y 7).
- **Hiperparámetros:** α=0.1, γ=0.99, ε 1.0→0.01 decay 0.999 (Decisión 6).
- **Dyna-Q:** aprende ~2× más rápido en episodios y es **más robusto** (Decisiones 4 y 8).
- **Modelo final `shaped_s5`:** verificado en **200 episodios → 100% éxito, reward 93.45
  ± 0.42** (rango 92.15–94.00), 148 pasos (Decisión 9).
- **Tiempos:** ~30-50 s por entrenamiento de Q-Learning (2000-5000 ep); Dyna-Q más lento.

## Estructura sugerida del informe (LOST)

1. Descripción del ambiente y por qué es difícil.
2. Discretización: por qué, alternativas y su impacto (Decisión 5).
3. Q-Learning: implementación y el problema del reward disperso + shaping (Decisión 3).
4. Hiperparámetros: grilla y elección justificada (Decisión 6).
5. Dyna-Q: implementación, comparación y robustez con semillas (Decisiones 4 y 8).
6. Modelo final elegido y verificación (Decisión 9).
7. Dificultades encontradas (óptimo local, grillas finas — Decisiones 3, 5, 7).
8. Conclusiones.
9. Uso de IA generativa.
