# Muestra en tiempo real el progreso del entrenamiento de la RNA.
# Uso:  powershell -ExecutionPolicy Bypass -File ver_entrenamiento.ps1
# (Ctrl+C para dejar de ver; el entrenamiento sigue en segundo plano)

$registro = Get-ChildItem "C:\Users\James\.local\share\opencode\shell\*\sh_*.out" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 |
    Where-Object { (Get-Content $_.FullName -Tail 200) -match "FASE|F1 |F2 |paso " } |
    Select-Object -First 1

if (-not $registro) {
    Write-Host "No se encontro ningun registro de entrenamiento activo."
} else {
    Write-Host "Siguiendo el registro: $($registro.FullName)"
    Write-Host "(Ctrl+C para salir)"
    Get-Content $registro.FullName -Tail 25 -Wait
}
