# Procesamiento de Imágenes SAR para Detección de Erosión Costera

Proyecto desarrollado para el curso **"Python para HPC - Introducción a la programación HPC con Python y sus aplicaciones al campo de proceso de imágenes"**.

Este repositorio implementa y compara diferentes técnicas de procesamiento paralelo aplicadas al análisis de imágenes SAR (Synthetic Aperture Radar) para la detección y monitoreo de erosión costera.

---

## 📋 Descripción del proyecto

El proyecto implementa un pipeline completo de procesamiento de imágenes SAR que incluye:
- Filtrado de ruido speckle
- Mejora de contraste (CLAHE/Ecualización)
- Detección de bordes (Sobel/Canny)
- Segmentación tierra/agua (Otsu + morfología)
- Extracción de línea de costa
- Análisis de textura (varianza local)

### Técnicas de paralelización implementadas

1. **Secuencial OpenCV**: Implementación base sin paralelización
2. **Multiprocessing Pool**: Paralelización CPU usando múltiples procesos
3. **ThreadPoolExecutor**: Paralelización CPU usando hilos
4. **MPI (Message Passing Interface)**: Procesamiento distribuido con MPI4Py
5. **CuPy GPU**: Aceleración mediante procesamiento en GPU con CUDA

---

## 🛰️ Fuente de datos

### Dataset de imágenes SAR de Alaska

Las imágenes utilizadas en este proyecto provienen del producto **ORI (Orthorectified Radar Image)** de **Intermap Technologies**, obtenidas a través del portal **EarthExplorer** del Servicio Geológico de Estados Unidos (USGS). Los datos fueron adquiridos mediante el sistema aerotransportado **STAR-3** de Radar de Apertura Sintética Interferométrico (IFSAR) sobre Alaska, entre el 23 de agosto y el 6 de septiembre de 2012.

#### Especificaciones de los datos originales

| Parámetro | Valor |
|-----------|-------|
| **Fuente** | USGS EarthExplorer - IFSAR ORI Alaska |
| **Producto** | Orthorectified Radar Image (ORI) |
| **Tecnología** | Interferometric Synthetic Aperture Radar (IFSAR) |
| **Sensor** | STAR-3 (aerotransportado) |
| **Resolución espacial** | 0.625 metros |
| **Formato original** | GeoTIFF de 8 bits |
| **Dimensiones originales** | ~23,344 × 46,728 píxeles |
| **Tamaño por archivo** | ~1 GB |
| **Sistema de coordenadas** | Albers Conical Equal Area (NAD83 CORS96) |
| **Área de cobertura** | Alaska, USA |
| **Período de adquisición** | 23 de agosto - 6 de septiembre de 2012 |
| **Total de imágenes** | 50 |

#### Procesamiento aplicado al dataset

Las imágenes originales fueron procesadas para optimizar su almacenamiento y uso en aplicaciones de procesamiento paralelo, reduciendo sus dimensiones mediante reescalado:

- **Dimensiones procesadas:** 2,000 × 4,000 píxeles
- **Factor de reducción:** ~11.7× en ancho, ~11.7× en alto
- **Proporción de aspecto:** Mantenida (1:2 aproximadamente)
- **Formato:** GeoTIFF de 8 bits

Esta reducción permitió generar un dataset más manejable (de ~1 GB a ~80 MB por imagen) manteniendo las características espaciales relevantes de las imágenes radar para el análisis de erosión costera.

#### Cita recomendada

```
Intermap Technologies Inc. (2012). IFSAR ORI Alaska - Orthorectified Radar Images.
Obtenido de USGS EarthExplorer. Datos de adquisición: Agosto-Septiembre 2012.
Accedido: [fecha de descarga].
```

#### Descarga del dataset

El dataset preprocesado (50 imágenes, 2000×4000 px) se descarga automáticamente al ejecutar:

```bash
make download-dataset
```

Las imágenes se almacenarán en el directorio `dataset/` en formato PNG.

---

## 🚀 Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/jorgeceferinovaldez/parallel-sar-processor.git
cd parallel-sar-processor
```

### 2. Crear entorno Conda con Python 3.11

```bash
# Crear entorno conda con Python 3.11
conda create -n parallel-sar-processor python=3.11

# Activar el entorno
conda activate parallel-sar-processor
```

### 3. Instalar dependencias

#### Instalación Básica (CPU)

```bash
make install
```

Este comando instalará las siguientes dependencias principales:
- OpenCV (procesamiento de imágenes)
- NumPy (arrays y operaciones numéricas)
- Pandas (análisis de datos)
- Matplotlib/Seaborn (visualización)
- MPI4Py (procesamiento distribuido)
- CuPy (procesamiento GPU - requiere CUDA)
- gdown (descarga de datasets)

#### Instalación para desarrollo (Opcional)

```bash
make install-dev
```

Incluye herramientas adicionales:
- Testing (pytest, pytest-cov)
- Formateo (black, isort)
- Linting (flake8, pylint)
- Type checking (mypy)
- Jupyter notebooks
- Profiling (line-profiler, memory-profiler)

### 4. Requisitos del sistema

#### MPI
Para usar el procesamiento con MPI, necesitas tener instalado `mpiexec` o `mpirun`:

**Ubuntu/Debian:**
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install openmpi openmpi-devel
```

#### CUDA Toolkit 12.x (requerido para CuPy GPU)

**IMPORTANTE**: CuPy requiere el CUDA Toolkit completo instalado y configurado en el sistema para tener soporte de GPU. Sin CUDA Toolkit, CuPy no funcionará.

**Requisitos:**
- GPU NVIDIA compatible con CUDA (Compute Capability ≥ 3.5)
- Driver NVIDIA actualizado (versión ≥ 525.60.13 para CUDA 12.x)
- CUDA Toolkit 12.x completo

##### Opción 1: Instalación de CUDA Toolkit mediante Conda (recomendado)

La forma más sencilla y segura es instalar CUDA Toolkit directamente en el entorno conda:

```bash
# Activar el entorno
conda activate parallel-sar-processor

# Instalar CUDA Toolkit 12.x completo en el entorno conda
conda install -c nvidia cuda-toolkit=12.6

# Instalar cudnn (opcional pero recomendado para mejor rendimiento)
conda install -c conda-forge cudnn

# Verificar la instalación
nvcc --version
```

**Ventajas de usar Conda:**
- ✅ No requiere permisos de administrador
- ✅ Aislado en el entorno virtual (no afecta el sistema)
- ✅ Fácil de desinstalar o actualizar
- ✅ Compatible con múltiples versiones de CUDA en diferentes entornos
- ✅ Instalación automática de todas las dependencias

##### Opción 2: Instalación de CUDA Toolkit a nivel de sistema

Si prefieres instalar CUDA a nivel de sistema (requiere permisos sudo):

**Ubuntu/Debian:**
```bash
# Descargar e instalar CUDA Toolkit 12.6
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-6

# Configurar variables de entorno (agregar a ~/.bashrc)
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Recargar configuración
source ~/.bashrc
```

**Fedora/RHEL:**
```bash
# Descargar e instalar CUDA Toolkit 12.6
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf clean all
sudo dnf install cuda-toolkit-12-6

# Configurar variables de entorno
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

##### Verificar instalación de CUDA y driver NVIDIA

```bash
# Verificar versión de CUDA Toolkit instalado
nvcc --version

# Verificar driver NVIDIA y GPUs disponibles
nvidia-smi

# Verificar que CuPy detecta la GPU (después de instalar dependencias)
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'CUDA version: {cp.cuda.runtime.runtimeGetVersion()}'); print(f'Device: {cp.cuda.Device(0).compute_capability}')"
```

**Salida esperada de `nvidia-smi`:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
...
```

**Notas importantes:**
- Si no tienes GPU NVIDIA, el proyecto funcionará igualmente con las técnicas CPU (OpenCV, Pool, Thread, MPI)
- La versión de CuPy en `requirements.txt` es `cupy-cuda12x`, compatible con CUDA 12.0-12.6
- Para otras versiones de CUDA, instala el paquete CuPy correspondiente:
  - CUDA 11.x: `cupy-cuda11x`
  - CUDA 12.x: `cupy-cuda12x`

### 5. Verificar instalación

```bash
make check
```

Este comando verificará:
- ✅ Python instalado correctamente
- ✅ MPI disponible
- ✅ CUDA Toolkit (si está instalado)
- ✅ CuPy funcionando con GPU

---

## 📦 Comandos make disponibles

El proyecto utiliza un `Makefile` para automatizar todas las tareas. A continuación se describen los comandos disponibles:

### Comandos de configuración

| Comando | Descripción |
|---------|-------------|
| `make help` | Muestra la ayuda con todos los comandos disponibles |
| `make setup` | Configuración inicial completa (install + check) |
| `make install` | Instala dependencias de producción desde `requirements.txt` |
| `make install-dev` | Instala dependencias de desarrollo desde `requirements-dev.txt` |
| `make check` | Verifica requisitos del sistema (Python, MPI, CUDA, CuPy) |

### Comandos de procesamiento

| Comando | Descripción |
|---------|-------------|
| `make download-dataset` | Descarga el dataset de imágenes SAR desde Google Drive |
| `make MPI` | Ejecuta procesamiento con MPI (12 procesos por defecto) |
| `make parallel` | Ejecuta procesamiento con técnicas paralelas (Pool, Thread, CuPy) |
| `make comparacion` | Genera gráficos y resumen comparativo de todas las técnicas |
| `make pipeline` | Ejecuta el pipeline completo (download + MPI + parallel + comparacion) |

### Comandos auxiliares

| Comando | Descripción |
|---------|-------------|
| `make clean` | Limpia archivos generados (output/*, __pycache__, etc.) |
| `make procesamiento` | Ejecuta solo MPI + parallel (sin descarga ni comparación) |
| `make run-mpi NPROCS=8` | Ejecuta MPI con número personalizado de procesos |

---

## 🎯 Flujo de trabajo típico

### Ejecución completa del pipeline

```bash
# 1. Configuración inicial (solo primera vez)
make setup

# 2. Ejecutar pipeline completo
make pipeline
```

El comando `make pipeline` ejecutará secuencialmente:
1. Descarga del dataset
2. Procesamiento MPI
3. Procesamiento paralelo (Pool, Thread, CuPy)
4. Generación de gráficos comparativos

### Ejecución paso a paso

```bash
# 1. Descargar dataset
make download-dataset

# 2. Procesar con MPI
make MPI

# 3. Procesar con técnicas paralelas
make parallel

# 4. Generar comparación
make comparacion
```

### Personalizar número de procesos MPI

Por defecto se utilizan 12 procesos. Para cambiar este valor:

```bash
# Temporal (para una ejecución)
make run-mpi NPROCS=8

# Permanente (editar Makefile)
# Cambiar la línea: MPI_PROCESSES := 12
```

---

## 📊 Resultados y salidas

### Estructura de directorios de salida

Al ejecutar el pipeline o los comandos de descarga/procesamiento, se generarán automáticamente los siguientes directorios:

```
dataset/                 # [CREADO EN DESCARGA] Imágenes SAR Reescaaladas manteniendo proporciones
├── imagen_001.png       # Imagen SAR 1 (2000x4000 píxeles)
├── imagen_002.png       # Imagen SAR 2
├── ...
└── imagen_050.png       # Imagen SAR 50
                         # Total: 50 imágenes IFSAR ORI de Alaska

output/                  # [CREADO EN PROCESAMIENTO] Resultados por técnica
├── sar_opencv/          # Resultados procesamiento secuencial
├── sar_pool/            # Resultados multiprocessing Pool
├── sar_thread/          # Resultados ThreadPoolExecutor
├── sar_mpi/             # Resultados MPI
└── sar_cupy_gpu/        # Resultados CuPy GPU

metrics/                 # [CREADO EN PROCESAMIENTO] Métricas CSV
├── metricas_tecnicas_sar.csv              # Métricas de técnicas paralelas
├── metricas_tecnica_mpi.csv               # Métricas de MPI
├── metricas_procesos_mpi.csv              # Métricas por proceso MPI
├── metricas_imagenes_opencv.csv           # Métricas por imagen (OpenCV)
├── metricas_imagenes_pool.csv             # Métricas por imagen (Pool)
├── metricas_imagenes_thread.csv           # Métricas por imagen (Thread)
├── metricas_imagenes_mpi.csv              # Métricas por imagen (MPI)
└── metricas_imagenes_cupy.csv             # Métricas por imagen (CuPy)

summary/                 # [CREADO EN COMPARACIÓN] Resumen y gráficos
├── comparacion_tecnicas_sar.png           # Gráficos comparativos
├── informe_final_sar.txt                  # Informe textual
└── metricas_todas_tecnicas_completo.csv   # Tabla comparativa completa
```

### Imágenes generadas por técnica

Para cada imagen SAR procesada se generan 5 salidas:

1. **enhanced.png**: Imagen con contraste mejorado (post-CLAHE/ecualización)
2. **edges.png**: Mapa de detección de bordes (Sobel)
3. **segmentation.png**: Segmentación binaria tierra/agua
4. **coastline.png**: Línea de costa extraída
5. **texture.png**: Mapa de análisis de textura (varianza local)

### Métricas calculadas

Para cada imagen se calculan las siguientes métricas:

| Métrica | Descripción |
|---------|-------------|
| `intensidad_media` | Intensidad promedio de la imagen mejorada |
| `intensidad_std` | Desviación estándar de la intensidad |
| `num_bordes_canny` | Número de píxeles de bordes detectados (Canny) |
| `intensidad_gradiente_media` | Intensidad promedio del gradiente (Sobel) |
| `porcentaje_tierra` | Porcentaje de píxeles clasificados como tierra |
| `porcentaje_agua` | Porcentaje de píxeles clasificados como agua |
| `num_contornos_costa` | Número de contornos de línea de costa detectados |
| `longitud_costa_px` | Longitud total de la línea de costa en píxeles |
| `rugosidad_media` | Rugosidad/variación de textura promedio |

---

## 🔄 Diagrama de flujo de procesamiento

El siguiente diagrama muestra el pipeline de procesamiento aplicado a cada imagen:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE PROCESAMIENTO SAR                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Imagen SAR     │  Entrada: Imagen en formato TIFF/PNG/JPG
│  (Original)     │  Ejemplo: imagen_costera_001.tif
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. CARGA Y      │  • Lectura en escala de grises
│    REDIMENSIÓN  │  • Redimensión a 1000x2000 píxeles (ancho x alto)
└────────┬────────┘  • Normalización de valores 0-255
         │
         ▼
┌─────────────────┐
│ 2. FILTRADO     │  • Filtro bilateral (reducción ruido speckle)
│    DE RUIDO     │  • Non-Local Means Denoising
└────────┬────────┘  • Preservación de bordes
         │
         ▼
┌─────────────────┐
│ 3. MEJORA DE    │  • OpenCV/Pool/Thread/MPI: CLAHE (Contrast Limited AHE)
│    CONTRASTE    │  • CuPy GPU: Ecualización de histograma en GPU
└────────┬────────┘  → Salida: enhanced.png
         │
         ├─────────────────────────────────────────────┐
         │                                             │
         ▼                                             ▼
┌─────────────────┐                          ┌─────────────────┐
│ 4. DETECCIÓN    │                          │ 5. SEGMENTACIÓN │
│    DE BORDES    │                          │    TIERRA/AGUA  │
├─────────────────┤                          ├─────────────────┤
│ • Canny (CPU)   │                          │ • Umbralización │
│ • Sobel X/Y     │                          │   Otsu          │
│ • Magnitud de   │                          │ • Morfología    │
│   gradiente     │                          │   (cierre +     │
└────────┬────────┘                          │    apertura)    │
         │                                   └────────┬────────┘
         │                                            │
         ▼                                            ▼
    edges.png                               segmentation.png
         │                                            │
         │                                            ▼
         │                                   ┌─────────────────┐
         │                                   │ 6. EXTRACCIÓN   │
         │                                   │    LÍNEA COSTA  │
         │                                   ├─────────────────┤
         │                                   │ • Detección de  │
         │                                   │   contornos     │
         │                                   │ • Filtrado por  │
         │                                   │   área >1000px  │
         │                                   │ • Cálculo de    │
         │                                   │   longitud      │
         │                                   └────────┬────────┘
         │                                            │
         │                                            ▼
         │                                      coastline.png
         │
         ▼
┌─────────────────┐
│ 7. ANÁLISIS     │  • Filtrado de varianza local (ventana 15x15)
│    DE TEXTURA   │  • Cálculo de rugosidad
└────────┬────────┘  • Identificación de zonas de alta variabilidad
         │
         ▼
    texture.png
         │
         │
         ▼
┌─────────────────┐
│ 8. CÁLCULO DE   │  • intensidad_media, intensidad_std
│    MÉTRICAS     │  • num_bordes, intensidad_gradiente_media
└────────┬────────┘  • % tierra/agua, num_contornos, longitud_costa
         │            • rugosidad_media
         ▼
┌─────────────────┐
│  SALIDA FINAL   │  5 imágenes procesadas + 9 métricas por imagen
│  (5 imágenes +  │
│   métricas CSV) │
└─────────────────┘

NOTAS:
━━━━━━
• Todas las técnicas (OpenCV, Pool, Thread, MPI, CuPy) siguen el mismo pipeline
• La diferencia está en cómo se paraleliza el procesamiento de múltiples imágenes
• CuPy ejecuta las operaciones 1-7 en GPU, luego transfiere a CPU para paso 8
• MPI distribuye las imágenes entre múltiples procesos de forma cíclica
• Pool/Thread procesan múltiples imágenes en paralelo usando cores CPU
```

---

## 🧪 Descripción de scripts principales

### `download_dataset.py`
Descarga automáticamente el dataset de imágenes SAR desde Google Drive.

**Funcionalidad:**
- Descarga archivo ZIP desde Google Drive
- Descomprime en directorio `dataset/`
- Limpia archivo temporal

### `coastal_SAR_parallel.py`
Implementa y compara tres técnicas de paralelización CPU + GPU:

**Técnicas implementadas:**
1. **Secuencial OpenCV**: Baseline sin paralelización
2. **Multiprocessing Pool**: `multiprocessing.Pool` con N procesos
3. **ThreadPoolExecutor**: `concurrent.futures.ThreadPoolExecutor`
4. **CuPy GPU**: Procesamiento acelerado en GPU

**Salidas:**
- Imágenes procesadas en `output/sar_{tecnica}/`
- Métricas en `metrics/metricas_{tecnica}.csv`

### `coastal_SAR_MPI.py`
Implementa procesamiento distribuido usando MPI4Py.

**Características:**
- Distribución cíclica de imágenes entre procesos
- Proceso maestro (rank 0) coordina y recopila resultados
- Sincronización con barreras MPI
- Cálculo de eficiencia y balance de carga

**Salidas:**
- Imágenes procesadas en `output/sar_mpi/`
- Métricas por proceso y globales en `metrics/`

### `graph_comparison.py`
Genera visualizaciones y resumen comparativo de todas las técnicas.

**Gráficos generados:**
1. Tiempo total de procesamiento
2. Speedup vs secuencial
3. Throughput (img/s)
4. Comparación de métricas de imagen (intensidad, segmentación, etc.)
5. Tabla comparativa completa

**Salidas:**
- `summary/comparacion_tecnicas_sar.png`
- `summary/informe_final_sar.txt`
- `summary/metricas_todas_tecnicas_completo.csv`

### `tools_img.py`
Módulo con la función principal de procesamiento SAR usando OpenCV.

**Función principal:**
```python
procesar_imagen_sar_costera(
    ruta_entrada: str,
    carpeta_salida: str,
    nombre_base: str,
    tamanio_fijo: tuple = (1000, 2000)
) -> dict
```

### `tools_cupy.py`
Módulo con la función de procesamiento SAR acelerado en GPU.

**Función principal:**
```python
procesar_imagen_sar_gpu(
    ruta_entrada: str,
    carpeta_salida: str,
    nombre_base: str,
    tamanio_fijo: tuple = (1000, 2000)
) -> dict
```

**Características:**
- Transferencia de datos CPU → GPU
- Operaciones en GPU con CuPy
- Sincronización y liberación de memoria GPU
- Transferencia de resultados GPU → CPU

---

## 🔧 Personalización

### Cambiar número de procesos/hilos

**En el Makefile (MPI):**
```makefile
MPI_PROCESSES := 12  # Cambiar este valor
```

**En `coastal_SAR_parallel.py`:**
```python
P = mp.cpu_count()              # Pool: todos los cores
num_hilos = P                   # Thread: todos los hilos
```

### Cambiar tamaño de imagen procesada

En los scripts de procesamiento:
```python
tamanio_fijo = (1000, 2000)  # (ancho, alto) en píxeles
```

### Agregar nuevas métricas

1. Modificar función `procesar_imagen_sar_costera()` en `tools_img.py`
2. Agregar cálculo de nueva métrica en el diccionario `metricas`
3. Actualizar `graph_comparison.py` para visualizar la nueva métrica

---

## 📈 Resultados obtenidos

### Benchmarks reales - 50 Imágenes SAR (1000x2000 px)

Los siguientes resultados fueron obtenidos procesando 50 imágenes SAR con el pipeline completo:

| Técnica | Tiempo (s) | Speedup | Velocidad (img/s) | Imágenes |
|---------|-----------|---------|-------------------|----------|
| Secuencial OpenCV | 18.60 | 1.00x | 2.69 | 50 |
| Pool (24 cores) | 10.23 | 1.82x | 4.89 | 50 |
| ThreadPool (24 threads) | 8.13 | 2.29x | 6.15 | 50 |
| MPI (12 cores) | 10.81 | 1.72x | 4.63 | 50 |
| CuPy GPU | 12.75 | 1.46x | 3.92 | 50 |

### Análisis de resultados

**🏆 Mejor rendimiento:** ThreadPool (24 threads) - 2.29x speedup
- Aprovecha mejor el paralelismo a nivel de hilos para operaciones I/O intensivas
- Menor overhead de comunicación entre hilos comparado con procesos

**🥈 Segundo lugar:** Pool (24 cores) - 1.82x speedup
- Buen balance entre rendimiento y simplicidad
- Mayor overhead por creación de procesos

**🥉 Tercer lugar:** MPI (12 cores) - 1.72x speedup
- Excelente para procesamiento distribuido en clusters
- Escalabilidad lineal al agregar más nodos

**💡 Nota sobre CuPy GPU (1.46x speedup):**
- El speedup relativamente bajo se debe a:
  - Overhead de transferencia CPU ↔ GPU
  - Imágenes de tamaño moderado (1000x2000 px)
  - Pipeline con muchas operaciones secuenciales
- CuPy GPU es más eficiente con:
  - Imágenes muy grandes (>4000x4000 px)
  - Procesamiento de muchas imágenes en batch
  - Operaciones matemáticas intensivas

**Hardware utilizado:**
- **CPU:** AMD Ryzen 9 5900X (12 cores / 24 threads)
- **RAM:** 32 GB
- **GPU:** NVIDIA GeForce RTX 4070 Ti Super (16 GB VRAM)
- **Almacenamiento:** SSD NVMe 2 TB
- **Tamaño de imagen:** 1000x2000 píxeles (fijo)
- **Pipeline:** 7 etapas de procesamiento por imagen

*Nota: Los valores pueden variar según el hardware utilizado. ThreadPool muestra el mejor rendimiento en este caso específico debido a la naturaleza I/O intensiva del procesamiento de imágenes con OpenCV.*

---

## ⚠️ Solución de problemas

### Error: MPI no encontrado
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
pip install mpi4py
```

### Error: CuPy no puede encontrar CUDA

**Síntoma:**
```python
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

**Causa:** CUDA Toolkit no está instalado o no está en el PATH.

**Solución 1 - Instalar CUDA Toolkit mediante Conda (Recomendado):**
```bash
conda activate parallel-sar-processor
conda install -c nvidia cuda-toolkit=12.6
```

**Solución 2 - Verificar instalación existente:**
```bash
# Verificar que CUDA esté instalado
nvcc --version

# Si está instalado pero no se encuentra, agregar al PATH
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# Hacer permanente agregando a ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Solución 3 - Verificar versión de CuPy:**
```bash
# Verificar versión de CUDA instalada
nvcc --version

# Reinstalar CuPy para la versión correcta de CUDA
pip uninstall cupy-cuda12x

# Para CUDA 11.x
pip install cupy-cuda11x

# Para CUDA 12.x
pip install cupy-cuda12x
```

### Error: CuPy instalado pero no detecta GPU

**Síntoma:**
```python
CuPyException: CUDA environment is not correctly set up
```

**Diagnóstico:**
```bash
# Verificar que nvidia-smi funcione
nvidia-smi

# Verificar que el driver NVIDIA esté cargado
lsmod | grep nvidia

# Probar importar CuPy y ver el error específico
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

**Solución:**
```bash
# Si nvidia-smi no funciona, reinstalar driver NVIDIA
sudo ubuntu-drivers autoinstall
sudo reboot

# Si el driver funciona pero CuPy no, reinstalar CUDA Toolkit
conda install -c nvidia cuda-toolkit=12.6 --force-reinstall
```

### Error: Memoria insuficiente en GPU

**Síntoma:**
```
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating XXX bytes
```

**Soluciones:**

1. **Reducir tamaño de imagen:**
```python
# En los scripts de procesamiento
tamanio_fijo = (800, 1600)  # Reducir de (1000, 2000)
```

2. **Liberar memoria GPU manualmente:**
```python
import cupy as cp
# Después de procesar cada imagen
cp.get_default_memory_pool().free_all_blocks()
```

3. **Verificar memoria disponible:**
```bash
nvidia-smi
# Buscar la columna "Memory-Usage"
```

4. **Cerrar otros procesos que usen GPU:**
```bash
# Ver procesos que usan GPU
nvidia-smi

# Si hay otros procesos, cerrarlos o reducir su uso
```

### Error: Timeout en descarga de dataset

Si `make download-dataset` falla, descargar manualmente desde:
```
https://drive.google.com/file/d/1vNYzal2fUyaZO5zwFmWgquj6f8ghLzab/view
```
Y extraer en el directorio `dataset/`.

### Error: ImportError al ejecutar scripts con MPI

**Síntoma:**
```
ImportError: No module named 'mpi4py'
```

**Solución:**
```bash
# Asegurarse de que mpi4py esté instalado en el entorno
conda activate parallel-sar-processor
pip install mpi4py

# Si persiste, reinstalar desde conda
conda install -c conda-forge mpi4py
```

### Advertencia: Driver NVIDIA vs CUDA Toolkit

**Es importante entender la diferencia:**

- **Driver NVIDIA**: Software a nivel de sistema que permite comunicación con la GPU
  - Se instala a nivel de sistema (requiere sudo)
  - Compatible con múltiples versiones de CUDA
  - Verifica con: `nvidia-smi`

- **CUDA Toolkit**: Librerías y herramientas de desarrollo CUDA
  - Puede instalarse por usuario (conda) o sistema (sudo)
  - Debe ser compatible con el driver instalado
  - Verifica con: `nvcc --version`

**Compatibilidad Driver-CUDA:**
- Driver ≥ 525.60.13 → soporta CUDA 12.x
- Driver ≥ 450.80.02 → soporta CUDA 11.x

```bash
# Ver versión máxima de CUDA soportada por el driver
nvidia-smi
# Buscar "CUDA Version: X.X" en la esquina superior derecha
```

---

## 📚 Dependencias principales

| Librería | Versión | Propósito |
|----------|---------|-----------|
| opencv-python | 4.12.0.88 | Procesamiento de imágenes |
| numpy | 2.2.6 | Operaciones numéricas |
| pandas | 2.3.2 | Análisis de datos |
| matplotlib | 3.10.6 | Visualización |
| seaborn | 0.13.2 | Gráficos estadísticos |
| mpi4py | 4.1.0 | Procesamiento distribuido MPI |
| cupy-cuda12x | 13.6.0 | Procesamiento GPU |
| gdown | 5.2.0 | Descarga desde Google Drive |

---

## 👤 Autor

Proyecto desarrollado por Jorge Ceferino Valdez.

---

## 📄 Licencia

Este proyecto es con fines académicos para el curso mencionado.
