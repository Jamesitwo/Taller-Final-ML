/**
 * 08 - Verificación de la implementación JavaScript de la CNN (Node.js)
 * ====================================================================
 * Compara la salida de la red implementada en web/red-neuronal.js
 * (TensorFlow.js) contra la salida esperada calculada con PyTorch y
 * guardada por 03_exportar_js.py en web/modelo/verificacion/.
 *
 * Uso:
 *   node entrenamiento/08_verificar_js_node.js
 *
 * Requiere TensorFlow.js para Node:
 *   npm install @tensorflow/tfjs
 * (o definir NODE_PATH apuntando a una instalación existente)
 */

const fs = require("fs");
const path = require("path");

function cargarTF() {
  try {
    return require("@tensorflow/tfjs");
  } catch (e) {
    const alternativa = path.join(
      process.env.TEMP || "/tmp", "opencode", "tfjs", "node_modules", "@tensorflow", "tfjs"
    );
    return require(alternativa);
  }
}

function aArrayBuffer(ruta) {
  const buf = fs.readFileSync(ruta);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

async function main() {
  const tf = cargarTF();
  await tf.ready();
  console.log("Backend de TensorFlow.js:", tf.getBackend());

  const raiz = path.join(__dirname, "..");
  const manifiesto = JSON.parse(fs.readFileSync(path.join(raiz, "web/modelo/manifiesto.json"), "utf8"));
  const pesos = aArrayBuffer(path.join(raiz, "web/modelo/pesos.bin"));
  const entrada = new Float32Array(aArrayBuffer(path.join(raiz, "web/modelo/verificacion/entrada.bin")));
  const esperado = JSON.parse(fs.readFileSync(path.join(raiz, "web/modelo/verificacion/salida_esperada.json"), "utf8"));

  // carga el mismo código que usará el navegador
  const codigo = fs.readFileSync(path.join(raiz, "web/red-neuronal.js"), "utf8");
  const fabrica = new Function("tf", "MANIFIESTO", codigo + "\n;return { RedNeuronalConvolucional };");
  const { RedNeuronalConvolucional } = fabrica(tf, manifiesto);

  const red = new RedNeuronalConvolucional(manifiesto, pesos);
  const x = tf.tensor4d(entrada, [1, manifiesto.entrada, manifiesto.entrada, 3], "float32");
  const t0 = Date.now();
  const y = red.propagar(x);
  const obtenido = Array.from(await y.data());
  console.log("Inferencia:", Date.now() - t0, "ms");

  const probs = esperado.probabilidades;
  let maxDif = 0, sumaDif = 0;
  for (let i = 0; i < probs.length; i++) {
    const d = Math.abs(obtenido[i] - probs[i]);
    maxDif = Math.max(maxDif, d);
    sumaDif += d;
  }
  const argmax = (a) => a.indexOf(Math.max(...a));
  const top5 = (a) => a.map((v, i) => [v, i]).sort((p, q) => q[0] - p[0]).slice(0, 5).map((p) => p[1]);
  const mismoArgmax = argmax(obtenido) === argmax(probs);
  const mismoTop5 = JSON.stringify(top5(obtenido)) === JSON.stringify(top5(probs));
  const aprobado = maxDif < 0.02 && mismoArgmax && mismoTop5;

  console.log("Máxima diferencia absoluta:", maxDif.toExponential(3));
  console.log("Diferencia media:", (sumaDif / probs.length).toExponential(3));
  console.log("Argmax idéntico:", mismoArgmax, "| clase:", esperado.clases[argmax(probs)]);
  console.log("Top-5 idéntico:", mismoTop5);
  console.log("Esperado:", JSON.stringify(esperado.top5));
  console.log("Obtenido:", JSON.stringify(top5(obtenido).map((i) => ({ clase: esperado.clases[i], probabilidad: +obtenido[i].toFixed(5) }))));
  console.log(aprobado ? "\n✅ VERIFICACIÓN APROBADA" : "\n❌ VERIFICACIÓN RECHAZADA");
  process.exit(aprobado ? 0 : 1);
}

main().catch((e) => { console.error(e); process.exit(1); });
