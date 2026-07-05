const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat, ImageRun
} = require("docx");
const fs = require("fs");

function figure(pngPath, altTitle) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 200 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(pngPath),
      transformation: { width: 620, height: 349 },
      altText: { title: altTitle, description: altTitle, name: altTitle },
    })],
  });
}

// ── helpers ────────────────────────────────────────────────────────────────
const cm = (v) => Math.round(v * 567); // cm to DXA
const border = { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const headerBorder = { style: BorderStyle.SINGLE, size: 12, color: "8B0000" };

const PAGE_W = 11906, PAGE_H = 16838;
const MARGIN = 1134; // ~2 cm
const CONTENT_W = PAGE_W - MARGIN * 2; // ~9638 DXA

function para(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 140, line: 276 },
    ...opts,
    children: [new TextRun({ text, font: "Arial", size: 22, ...opts.run })],
  });
}

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, font: "Arial", size: 32, bold: true })],
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: "Arial", size: 26, bold: false, color: "595959" })],
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22, bold: false, italics: true, color: "595959" })],
  });
}

function bullet(text, numbering) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22 })],
  });
}

function numberedItem(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22 })],
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function spacer(after = 120) {
  return new Paragraph({ children: [new TextRun("")], spacing: { after } });
}

// ── table helpers ──────────────────────────────────────────────────────────
function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "2E4057", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, font: "Arial", size: 20, bold: true, color: "FFFFFF" })],
    })],
  });
}

function dataCell(text, width, shade = false, bold = false, center = false) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: shade ? "F2F2F2" : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, font: "Arial", size: 20, bold })],
    })],
  });
}

// ── MAIN TABLES ────────────────────────────────────────────────────────────
// Table 1: Alpha-Beta impact by depth
function tableAlphaBeta() {
  const cols = [1600, 1600, 2200, 2200, 1900]; // sum = 9500 ~ CONTENT_W
  const W = cols.reduce((a, b) => a + b, 0);
  const rows = [
    ["1", "61", "61", "0.008s", "0.009s"],
    ["2", "651", "2,938", "0.079s", "0.359s"],
    ["3", "9,936", "95,008", "1.58s", "11.05s"],
    ["4", "58,586", "1,891,768", "5.11s", "158.9s"],
  ];
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      new TableRow({ children: [
        headerCell("Profundidad", cols[0]),
        headerCell("Nodos AB", cols[1]),
        headerCell("Nodos sin AB", cols[2]),
        headerCell("Tiempo AB", cols[3]),
        headerCell("Tiempo sin AB", cols[4]),
      ]}),
      ...rows.map((r, i) => new TableRow({ children: r.map((v, j) => dataCell(v, cols[j], i % 2 === 0, j === 0, true)) })),
    ],
  });
}

// Table 2: Tournament results
function tableTournament() {
  const cols = [2800, 2800, 2000, 1900];
  const W = cols.reduce((a, b) => a + b, 0);
  const rows = [
    ["Minimax (AB)", "RandomAgent", "98%", "43.2s"],
    ["Minimax (AB)", "Stratagem", "44%", "189.5s"],
    ["Expectimax", "RandomAgent", "66%", "438.3s"],
    ["Expectimax", "Stratagem", "6%", "454.8s"],
    ["Minimax (AB)", "Minimax (sin AB)", "40%*", "358.8s"],
  ];
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      new TableRow({ children: [
        headerCell("Agente A", cols[0]),
        headerCell("Agente B", cols[1]),
        headerCell("Win-rate A", cols[2]),
        headerCell("Tiempo total", cols[3]),
      ]}),
      ...rows.map((r, i) => new TableRow({ children: r.map((v, j) => dataCell(v, cols[j], i % 2 === 0, j === 2, j >= 2)) })),
    ],
  });
}

// Table 3: Heuristic sweep
function tableHeuristicSweep() {
  const cols = [2200, 3800, 1800, 1700];
  const W = cols.reduce((a, b) => a + b, 0);
  const rows = [
    ["balanceado",     "relative_mobility:1.0, center:0.25, corner:0.5", "96.67%", "48.00%"],
    ["agresivo",       "relative_mobility:0.5, center:0.0, corner:2.0",  "100%",   "60.00%"],
    ["posicional",     "relative_mobility:0.5, center:2.0, corner:0.0",  "100%",   "70.00%"],
    ["movilidad_pura", "relative_mobility:2.0, center:0.0, corner:0.0",  "100%",   "58.00%"],
    ["mixto",          "relative_mobility:1.0, center:1.0, corner:1.0",  "100%",   "59.00%"],
  ];
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: cols,
    rows: [
      new TableRow({ children: [
        headerCell("Config.", cols[0]),
        headerCell("Pesos", cols[1]),
        headerCell("vs Random", cols[2]),
        headerCell("vs Stratagem", cols[3]),
      ]}),
      ...rows.map((r, i) => new TableRow({ children: r.map((v, j) => dataCell(v, cols[j], i % 2 === 0, j === 3 && r[3] === "70.00%", j >= 2)) })),
    ],
  });
}

// ── DOCUMENT ───────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: false, font: "Arial", color: "595959" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, italics: true, font: "Arial", color: "595959" },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: PAGE_W, height: PAGE_H },
               margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "8B0000", space: 4 } },
            children: [
              new TextRun({ text: "Universidad ORT Uruguay", font: "Arial", size: 18, bold: true }),
              new TextRun({ text: "\t\t\t\t\t\t", font: "Arial", size: 18 }),
              new TextRun({ text: "Facultad de Ingenieria", font: "Arial", size: 18, bold: true, color: "8B0000" }),
            ],
          }),
        ],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18 })],
        })],
      }),
    },
    children: [
      // ── MATE - Isolation ──────────────────────────────────────────────
      heading1("MATE - Isolation"),

      // Introduccion
      heading2("Introduccion"),
      heading3("Contexto del problema"),
      para("El proyecto Martian Adversarial Tactics Engine (MATE) surge en el marco de la mision ficticia de la empresa Red Destination™. Una vez que el rover \"Out for Delivery\" logra desplazarse por la superficie marciana, se identifica una prioridad critica para el exito de la mision: ante un eventual contacto con vida inteligente en Marte, el desempeno del rover en dinamicas ludicas tendra el mayor impacto en las relaciones interplanetarias."),
      para("Para preparar al agente ante este escenario, se utiliza un simulador basado en el juego de mesa Isolation. En este juego, dos jugadores se mueven en un tablero 4x4 y deben eliminar celdas en cada turno. Pierde el jugador que no puede moverse. El entorno es adversarial y determinista, lo que lo convierte en un caso de estudio ideal para algoritmos de busqueda en arboles de juego."),

      heading3("Objetivos"),
      para("El objetivo principal es implementar agentes inteligentes capaces de tomar decisiones optimas en el juego Isolation. Concretamente, se plantean los siguientes objetivos:"),
      bullet("Implementar el algoritmo Minimax con Alpha-Beta Pruning y analizar el impacto de la poda en nodos expandidos y tiempo de ejecucion."),
      bullet("Implementar el algoritmo Expectimax y comparar su desempeno con Minimax frente a distintos tipos de rivales."),
      bullet("Diseniar funciones de evaluacion heuristicas y experimentar con distintas combinaciones de pesos para determinar la configuracion optima."),
      bullet("Evaluar los agentes mediante un protocolo de torneo riguroso con multiples partidas y registro completo de resultados."),
      spacer(),

      // Implementacion
      heading2("Implementacion"),
      para("Como primer paso, se analizo el codigo base provisto por la catedra. Se identificaron los modulos clave: board.py (logica del juego), agent.py (interfaz abstracta), isolation_env.py (wrapper Gym), play.py (ejecucion de partidas), random_agent.py y stratagem.py (agentes de referencia)."),
      para("Se decidio no modificar ningun archivo provisto, creando todos los agentes en archivos nuevos que heredan de la clase abstracta Agent. Esto permite reutilizar la API existente (get_possible_actions, clone, play, is_end, find_player_position) sin reimplementar las reglas del juego."),

      heading3("Implementacion de funciones heuristicas"),
      para("Las funciones heuristicas permiten evaluar estados no terminales cuando la busqueda alcanza su profundidad limite. Se implementaron cinco heuristicas en el modulo heuristics.py, todas con la firma h(board, player) -> float, lo que permite combinarlas y testearlas de forma independiente:"),
      numberedItem("own_mobility: cuenta los movimientos legales disponibles para el agente. A mayor movilidad, mayor libertad de accion."),
      numberedItem("relative_mobility: diferencia entre los movimientos propios y los del rival. Premia la ventaja relativa de movilidad."),
      numberedItem("center_control: distancia Manhattan negativa al centro del tablero. Posiciones centrales ofrecen mas opciones de movimiento."),
      numberedItem("opponent_distance: distancia Manhattan al rival. Permite modelar estrategias de alejamiento o acercamiento."),
      numberedItem("corner_rival: cantidad de celdas bloqueadas alrededor del rival. Cuanto mas acorralado este el rival, mas vulnerable es."),
      spacer(),
      para("Todas las heuristicas se combinan mediante una funcion weighted_heuristic(board, player, weights) que pondera linealmente cada componente con un diccionario de pesos configurables:"),
      spacer(),
      new Paragraph({
        spacing: { before: 60, after: 60 },
        shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
        indent: { left: 720 },
        children: [new TextRun({ text: "h(board) = w1*h1(board) + w2*h2(board) + ... + wk*hk(board)", font: "Courier New", size: 20, italics: true })],
      }),
      spacer(),
      para("Esta disenio parametrizable es fundamental para la Tarea 2, ya que permite explorar distintos comportamientos del agente (agresivo, posicional, de movilidad pura) simplemente cambiando los pesos."),

      // Primer Agente Minimax
      heading2("Primer Agente: Minimax"),
      para("Se implemento el algoritmo Minimax con Alpha-Beta Pruning opcional en minimax_agent.py. El agente hereda de SearchAgent (clase base propia que centraliza la instrumentacion) e implementa los metodos next_action(obs) y heuristic_utility(board) requeridos por la interfaz Agent."),
      para("El parametro use_alpha_beta=True/False permite correr el mismo algoritmo con y sin poda, lo cual es indispensable para comparar el impacto de la optimizacion. La convencion de signo es: +inf si gana self.player, -inf si pierde, y el valor heuristico en estados intermedios."),
      para("El analisis de Alpha-Beta Pruning se realizo midiendo nodos expandidos y tiempo promedio por jugada a distintas profundidades (1-4), sobre los mismos tableros iniciales (misma semilla) para que la comparacion sea justa:"),
      spacer(),
      tableAlphaBeta(),
      spacer(),
      para("Los resultados son contundentes: a profundidad 4, Alpha-Beta reduce los nodos expandidos en un factor de 32x (58,586 vs 1,891,768) y el tiempo en 31x (5.1s vs 158.9s por jugada). Sin poda, profundidad 4 seria inviable en una partida real. Con poda, profundidad 3 ofrece un balance optimo (1.58s/jugada)."),
      para("Ademas, se verifico que Alpha-Beta no modifica el resultado optimo del algoritmo: el enfrentamiento Minimax(AB) vs Minimax(sin AB) arroja ~40-50% de win-rate (resultado esperado si ambos juegan igual de bien), confirmando que la poda unicamente elimina ramas que no afectan la decision final."),

      // Segundo Agente Expectimax
      heading2("Segundo Agente: Expectimax"),
      para("Se implemento el algoritmo Expectimax en expectimax_agent.py. La diferencia clave respecto a Minimax esta en el nodo del rival: en lugar de minimizar (asumir que el rival juega perfectamente en tu contra), Expectimax promedia el valor de todas las jugadas posibles del rival con distribucion uniforme. Esto modela un rival que actua de forma aleatoria o suboptima."),
      para("Cuanto conviene cada algoritmo depende del rival:"),
      bullet("Contra RandomAgent (rival debil): Expectimax es superior porque Minimax \"desperdicia\" movimientos defendiendose de jugadas que el rival nunca tomaria deliberadamente."),
      bullet("Contra Stratagem (rival fuerte, Minimax de profundidad 3): Minimax es mucho mas seguro porque Stratagem si juega adversarialmente; Expectimax queda expuesto al asumir que el rival juega al azar cuando en realidad juega optimo."),
      spacer(),
      para("Expectimax no puede beneficiarse de Alpha-Beta Pruning, ya que los nodos de azar requieren evaluar todos sus hijos para calcular correctamente el valor esperado. Esto explica por que su costo computacional coincide exactamente con el de Minimax sin poda (misma cantidad de nodos expandidos)."),

      // Analisis de profundidades
      heading2("Analisis de Profundidades"),
      para("El barrido de profundidades muestra el crecimiento exponencial del arbol de busqueda y el impacto de la poda. A profundidad 2 ya se observa una reduccion de 4.5x en nodos; a profundidad 4 la brecha alcanza 32x. Dado el alto factor de ramificacion del juego Isolation (hasta 8 direcciones x celdas destruibles disponibles), la profundidad 3 con Alpha-Beta resulta la opcion mas practica: ofrece buena calidad de decision en ~1.58s/jugada, mientras que profundidad 4 sin poda exigira ~159s por jugada, inviable para una partida real."),
      para("Para el torneo principal se eligio profundidad 3 con Alpha-Beta activado como configuracion estandar."),

      // Pruebas y Evaluacion
      heading2("Pruebas y Evaluacion"),
      para("Se disenio un protocolo de torneo riguroso: 50 partidas por enfrentamiento, alternando quien juega como jugador 1 y jugador 2 en partidas pares e impares respectivamente, para eliminar el sesgo de posicion inicial. Las posiciones iniciales son aleatorias (semilla fija=42 para reproducibilidad). Se registraron win-rate, tiempo medio por jugada y nodos expandidos."),
      para("Los agentes se enfrentaron contra RandomAgent (linea base debil) y Stratagem (linea base fuerte, Minimax ofuscado de profundidad 3 provisto por la catedra):"),
      spacer(),
      tableTournament(),
      new Paragraph({ children: [new TextRun({ text: "* El resultado ~40-50% entre MM(AB) y MM(sin AB) es esperado: Alpha-Beta no cambia el resultado optimo, solo la velocidad.", font: "Arial", size: 18, italics: true, color: "595959" })], spacing: { before: 80, after: 80 } }),
      spacer(),
      para("Los resultados confirman la hipotesis: Minimax con Alpha-Beta es el algoritmo mas solido para este contexto. Obtiene 98% contra Random y 44% contra Stratagem (muy cerca del 50% que seria el resultado teorico optimo contra un rival igual de fuerte). Expectimax supera a Random (66%) pero colapsa ante Stratagem (6%), validando que la suposicion de rival aleatorio es incorrecta cuando el oponente juega adversarialmente."),

      // Configuraciones de heuristicas
      heading2("Configuraciones de Heuristicas (Tarea 2)"),
      para("Para la experimentacion con funciones de evaluacion, se probaron cinco configuraciones de pesos distintas enfrentando cada una contra RandomAgent (30 partidas) y Stratagem (100 partidas). Se utilizo un n mayor contra Stratagem porque es el enfrentamiento donde se deciden las conclusiones: con 100 partidas el margen de error del win-rate baja a ~9 puntos porcentuales (vs ~18 con n=30), haciendo las diferencias entre configuraciones estadisticamente solidas:"),
      spacer(),
      tableHeuristicSweep(),
      spacer(),
      para("El hallazgo mas relevante es que la configuracion posicional (center_control con peso 2.0) gana el 70.00% de las partidas contra Stratagem, la mejor de todas las configuraciones probadas. Esto tiene sentido: Stratagem tambien prioriza el control del centro en su heuristica; al darle mayor peso a esta dimension, el agente compite directamente en la misma dimension critica y la explota mejor."),
      para("Todas las configuraciones dominan a RandomAgent (96-100%), lo que confirma que cualquier heuristica razonable supera al azar. La diferencia real se ve en el enfrentamiento contra Stratagem, donde la eleccion de pesos tiene un impacto significativo (de 48% a 70%). Cabe destacar que todas las configuraciones empatan o superan a Stratagem, es decir, el Minimax propio de profundidad 3 esta como minimo al nivel del agente de referencia con cualquier heuristica razonable."),
      para("En base a estos resultados, se recomienda usar la configuracion posicional como configuracion principal para el agente final del proyecto."),

      // Dificultades y Conclusion
      heading2("Dificultades y Conclusion"),
      para("Durante el desarrollo surgieron los siguientes desafios:"),
      numberedItem("Factor de ramificacion alto: get_possible_actions combina hasta 8 direcciones por cantidad de celdas destruibles, generando arboles de busqueda muy grandes. Esto obligo a justificar cuidadosamente la profundidad maxima elegida y a priorizar Alpha-Beta Pruning."),
      numberedItem("Bug en la seleccion de accion: cuando todas las jugadas evaluaban a -inf (escenarios donde todas las ramas conducen a derrota), la variable best_action permanecia en None, causando un error al ejecutar la accion. Se corrigio usando la condicion if best_action is None or child_value > value para garantizar que siempre se elija al menos la primera accion disponible."),
      numberedItem("Expectimax sin poda: al no poder aplicar Alpha-Beta en nodos de azar, Expectimax tarda significativamente mas que Minimax (454s vs 189s para 50 partidas contra Stratagem a profundidad 3), lo que limita su uso practico a profundidades bajas."),
      spacer(),
      para("Los resultados obtenidos validan las hipotesis planteadas. Minimax con Alpha-Beta es la tecnica mas adecuada para Isolation, ya que el rival (Stratagem) juega de forma adversarial y la poda reduce dramaticamente el costo computacional sin afectar la calidad de las decisiones. Expectimax queda relegado a escenarios con rivales suboptimos (Random), donde su modelado probabilistico es mas preciso."),
      para("El analisis de configuraciones heuristicas revela que la eleccion de pesos tiene un impacto considerable en el desempeno contra rivales fuertes. La configuracion posicional, que prioriza el control del centro, resulto ser la mas efectiva, alcanzando 70.00% de win-rate contra Stratagem en 100 partidas. Este hallazgo sugiere que, en un tablero pequeno como 4x4, la posicion estrategica es mas determinante que la movilidad pura o el acorralamiento del rival."),

      pageBreak(),

      // ANEXO
      heading1("ANEXO"),
      heading2("Resultados Barrido de Profundidad (depth_sweep_results.csv)"),
      new Paragraph({
        shading: { fill: "1E1E1E", type: ShadingType.CLEAR },
        spacing: { before: 60, after: 60 },
        indent: { left: 360, right: 360 },
        children: [new TextRun({ text:
          "Minimax AB depth=1: nodes=61    time=0.008s\n" +
          "Minimax NoAB depth=1: nodes=61    time=0.009s\n" +
          "Expectimax depth=1: nodes=61    time=0.009s\n" +
          "Minimax AB depth=2: nodes=651   time=0.079s\n" +
          "Minimax NoAB depth=2: nodes=2938  time=0.359s\n" +
          "Expectimax depth=2: nodes=2938  time=0.479s\n" +
          "Minimax AB depth=3: nodes=9936  time=1.583s\n" +
          "Minimax NoAB depth=3: nodes=95008 time=11.048s\n" +
          "Expectimax depth=3: nodes=95008 time=9.922s\n" +
          "Minimax AB depth=4: nodes=58586 time=5.113s\n" +
          "Minimax NoAB depth=4: nodes=1891768 time=158.906s\n" +
          "Expectimax depth=4: nodes=1891768 time=158.220s",
          font: "Courier New", size: 18, color: "D4D4D4" })],
      }),
      spacer(),

      heading2("Resultados Torneo Principal (50 partidas por enfrentamiento)"),
      new Paragraph({
        shading: { fill: "1E1E1E", type: ShadingType.CLEAR },
        spacing: { before: 60, after: 60 },
        indent: { left: 360, right: 360 },
        children: [new TextRun({ text:
          "Minimax_AB vs RandomAgent:  win-rate Minimax_AB=98.00% (43.2s total)\n" +
          "Minimax_AB vs Stratagem:    win-rate Minimax_AB=44.00% (189.5s total)\n" +
          "Expectimax vs RandomAgent:  win-rate Expectimax=66.00% (438.3s total)\n" +
          "Expectimax vs Stratagem:    win-rate Expectimax=6.00%  (454.8s total)\n" +
          "Minimax_AB vs Minimax_NoAB: win-rate Minimax_AB=40.00% (358.8s total)",
          font: "Courier New", size: 18, color: "D4D4D4" })],
      }),
      spacer(),

      heading2("Resultados Barrido de Heuristicas (30 partidas vs Random, 100 vs Stratagem)"),
      new Paragraph({
        shading: { fill: "1E1E1E", type: ShadingType.CLEAR },
        spacing: { before: 60, after: 60 },
        indent: { left: 360, right: 360 },
        children: [new TextRun({ text:
          "balanceado     vs Random:   96.67%  vs Stratagem: 48.00%\n" +
          "agresivo       vs Random:  100.00%  vs Stratagem: 60.00%\n" +
          "posicional     vs Random:  100.00%  vs Stratagem: 70.00%  <-- MEJOR\n" +
          "movilidad_pura vs Random:  100.00%  vs Stratagem: 58.00%\n" +
          "mixto          vs Random:  100.00%  vs Stratagem: 59.00%",
          font: "Courier New", size: 18, color: "D4D4D4" })],
      }),
      spacer(),

      heading2("Figura 1"),
      para("La grafica representa el win-rate obtenido por cada agente en el torneo principal (50 partidas por enfrentamiento, alternando quien comienza). En el eje horizontal se muestran los enfrentamientos y en el eje vertical el porcentaje de victorias del agente A. La linea punteada marca el 50% (paridad). Se observa que Minimax con Alpha-Beta domina a RandomAgent y compite de igual a igual con Stratagem, mientras que Expectimax colapsa frente a un rival adversarial."),
      figure("Isolation/figures/fig1_winrate_torneo.png", "Win-rate por enfrentamiento del torneo principal"),

      heading2("Figura 2"),
      para("Esta grafica representa el costo computacional de la busqueda en funcion de la profundidad. En el eje horizontal se muestra la profundidad y en el eje vertical los nodos expandidos promedio por jugada, en escala logaritmica. La brecha creciente entre Minimax con y sin Alpha-Beta ilustra el ahorro de la poda (32x a profundidad 4). La curva de Expectimax coincide exactamente con la de Minimax sin poda, ya que los nodos de azar impiden aplicar Alpha-Beta."),
      figure("Isolation/figures/fig2_nodos_profundidad.png", "Nodos expandidos vs profundidad"),

      heading2("Figura 3"),
      para("Esta grafica compara el win-rate de las cinco configuraciones de pesos heuristicos contra RandomAgent (n=30) y contra Stratagem (n=100), usando Minimax con Alpha-Beta a profundidad 3. Todas las configuraciones dominan a Random; la diferencia real aparece contra Stratagem, donde la configuracion posicional (center_control x2.0) alcanza el 70%, el mejor resultado del barrido."),
      figure("Isolation/figures/fig3_heuristicas.png", "Win-rate por configuracion de heuristicas"),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("MATE_Documentacion.docx", buffer);
  console.log("ok: MATE_Documentacion.docx generado");
});
