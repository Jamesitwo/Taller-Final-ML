/**
 * Servidor estático mínimo para publicar la aplicación en Railway
 * (sin dependencias: solo la biblioteca estándar de Node).
 *
 * Sirve el contenido de la carpeta web/ en el puerto que asigne Railway.
 *   node server.js          -> http://localhost:3000
 *   PORT=8080 node server.js
 */

const http = require("http");
const fs = require("fs");
const path = require("path");

const PUERTO = process.env.PORT || 3000;
const RAIZ = path.join(__dirname, "web");

const TIPOS = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".bin": "application/octet-stream",
};

const servidor = http.createServer((peticion, respuesta) => {
  let ruta = decodeURIComponent(peticion.url.split("?")[0]);
  if (ruta === "/") ruta = "/index.html";

  const archivo = path.normalize(path.join(RAIZ, ruta));
  if (!archivo.startsWith(RAIZ)) {
    respuesta.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
    return respuesta.end("Prohibido");
  }

  fs.readFile(archivo, (error, datos) => {
    if (error) {
      respuesta.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      return respuesta.end("No encontrado");
    }
    respuesta.writeHead(200, {
      "Content-Type": TIPOS[path.extname(archivo).toLowerCase()] || "application/octet-stream",
      "Cache-Control": "public, max-age=3600",
    });
    respuesta.end(datos);
  });
});

servidor.listen(PUERTO, () => {
  console.log(`🌿 Identificador de Hojas listo en http://localhost:${PUERTO}`);
});
