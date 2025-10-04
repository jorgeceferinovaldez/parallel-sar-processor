# Makefile para Proyecto de Procesamiento de Imágenes SAR
# ========================================================

# Variables de configuración
PYTHON := python
MPI_PROCESSES := 12
MPI_EXEC := mpiexec

# Archivos de scripts
DOWNLOAD_SCRIPT := download_dataset.py
MPI_SCRIPT := coastal_SAR_MPI.py
PARALLEL_SCRIPT := coastal_SAR_parallel.py
COMPARISON_SCRIPT := graph_comparison.py

# Directorios (ajustar según tu estructura)
OUTPUT_DIR := output
GRAPHS_DIR := graphs

# Targets principales
.PHONY: all help download-dataset MPI parallel comparacion pipeline clean

# Target por defecto: mostrar ayuda
all: help

# Ayuda: muestra todos los comandos disponibles
help:
	@echo "=================================================="
	@echo "  Pipeline de Procesamiento de Imágenes SAR"
	@echo "=================================================="
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make download-dataset  - Descargar el dataset"
	@echo "  make MPI              - Procesar usando MPI ($(MPI_PROCESSES) procesos)"
	@echo "  make parallel         - Procesar usando técnicas paralelas"
	@echo "  make comparacion      - Generar gráficos y resumen comparativos"
	@echo "  make pipeline         - Ejecutar pipeline completo (todos los pasos)"
	@echo "  make clean            - Limpiar archivos generados"
	@echo "  make help             - Mostrar esta ayuda"
	@echo ""

# Descargar dataset
download-dataset:
	@echo "==> Descargando dataset..."
	$(PYTHON) $(DOWNLOAD_SCRIPT)
	@echo "==> Dataset descargado exitosamente "

# Procesamiento con MPI
MPI:
	@echo "==> Ejecutando procesamiento MPI con $(MPI_PROCESSES) procesos..."
	$(MPI_EXEC) -n $(MPI_PROCESSES) $(PYTHON) $(MPI_SCRIPT)
	@echo "==> Procesamiento MPI completado "

# Procesamiento paralelo
parallel:
	@echo "==> Ejecutando procesamiento paralelo..."
	$(PYTHON) $(PARALLEL_SCRIPT)
	@echo "==> Procesamiento paralelo completado "

# Generar comparación y gráficos
comparacion:
	@echo "==> Generando gráficos y comparaciones..."
	$(PYTHON) $(COMPARISON_SCRIPT)
	@echo "==> Gráficos generados exitosamente "

# Pipeline completo: ejecutar todos los pasos en orden
pipeline: download-dataset MPI parallel comparacion
	@echo ""
	@echo "=================================================="
	@echo "    Pipeline completo ejecutado exitosamente"
	@echo "=================================================="

# Limpieza de archivos generados
clean:
	@echo "==> Limpiando archivos generados..."
	rm -rf $(OUTPUT_DIR)/* $(GRAPHS_DIR)/* __pycache__ *.pyc
	@echo "==> Limpieza completada  "

# Targets adicionales útiles

# Ejecutar solo procesamiento (MPI + Paralelo)
.PHONY: procesamiento
procesamiento: MPI parallel
	@echo "==> Procesamiento completado  "

# Verificar requisitos del sistema
.PHONY: check
check:
	@echo "==> Verificando requisitos del sistema..."
	@which $(PYTHON) > /dev/null || (echo "Error: Python no encontrado" && exit 1)
	@which $(MPI_EXEC) > /dev/null || (echo "Error: MPI no encontrado" && exit 1)
	@echo "==> Verificación completada  "

# Ejecutar con número personalizado de procesos MPI
# Uso: make run-mpi NPROCS=8
.PHONY: run-mpi
run-mpi:
	@echo "==> Ejecutando MPI con $(or $(NPROCS),$(MPI_PROCESSES)) procesos..."
	$(MPI_EXEC) -n $(or $(NPROCS),$(MPI_PROCESSES)) $(PYTHON) $(MPI_SCRIPT)
