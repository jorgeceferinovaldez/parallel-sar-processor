import os
# Para evitar errores de OpenCV con imágenes grandes (> 2^24 píxeles)
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**40)

import time

# Librerías de paralelización para CPU
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tools_img import procesar_imagen_sar_costera
from tools_cupy import procesar_imagen_sar_gpu

# wrapper para multiprocessing
def procesar_wrapper(args):
    """
    Función wrapper para procesamiento paralelo de imágenes SAR costeras.

    Args:
        args (tuple): Una tupla que contiene tres elementos:
            - ruta_entrada (str): Ruta al archivo de imagen SAR de entrada
            - carpeta_salida (str): Ruta del directorio de salida para los resultados procesados
            - nombre_base (str): Nombre base para los archivos de salida

    Returns:
        Resultado de procesar_imagen_sar_costera: La salida de la función de procesamiento de imágenes SAR costeras
    """
    ruta_entrada, carpeta_salida, nombre_base = args
    return procesar_imagen_sar_costera(ruta_entrada, carpeta_salida, nombre_base)

if __name__ == '__main__':
    folder_in = 'dataset'
    folder_out = 'output'
    folder_metrics = 'metrics'
    
    # Directorios de salida para cada técnica
    folder_out_opencv = os.path.join(folder_out, 'sar_opencv')
    folder_out_pool = os.path.join(folder_out, 'sar_pool')
    folder_out_thread = os.path.join(folder_out, 'sar_thread')
    folder_out_cupy = os.path.join(folder_out, 'sar_cupy_gpu')
    
    os.makedirs(folder_out, exist_ok=True)
    os.makedirs(folder_out_opencv, exist_ok=True)
    os.makedirs(folder_out_pool, exist_ok=True)
    os.makedirs(folder_out_thread, exist_ok=True)
    os.makedirs(folder_out_cupy, exist_ok=True)
    os.makedirs(folder_metrics, exist_ok=True)
    
    # Preparamos lista de archivos
    archivos_lista = []
    for filename in os.listdir(folder_in):
        if filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            archivos_lista.append(filename)
    
    print("\n" + "="*80)
    print("Comparación de técnicas paralelas - Procesamiento SAR completo")
    print("="*80)
    print(f"Total de archivos: {len(archivos_lista)}")
    print("Tamaño fijo de procesamiento: 1000x2000 píxeles (ancho x alto)")
    print("Pipeline: Filtrado -> CLAHE -> Bordes -> Segmentación -> Costa -> Textura")
    print("="*80 + "\n")
    
    # creamos dataframes para las métricas
    metricas_tecnicas = []
    todas_metricas_imagenes = {'opencv': [], 'pool': [], 'thread': [], 'cupy': []}
    
    # 1. Usando técnica Opencv secuencial
    print("="*60)
    print("1. Procesamiento Secuencial con OpenCV...")
    print("="*60)
    
    start = time.time() # tiempo inicial de la técnica secuencial OpenCV
    
    for filename in archivos_lista:
        nombre_base = os.path.splitext(filename)[0]
        imagen_entrada = os.path.join(folder_in, filename)
        metricas = procesar_imagen_sar_costera(imagen_entrada, folder_out_opencv, nombre_base)
        if metricas:
            todas_metricas_imagenes['opencv'].append(metricas)
    
    end = time.time() # tiempo final de la técnica secuencial OpenCV
    tiempo_opencv = end - start
    velocidad_opencv = len(todas_metricas_imagenes['opencv']) / tiempo_opencv
    
    print(f" - Tiempo: {tiempo_opencv:.2f}s | Velocidad: {velocidad_opencv:.2f} img/s")
    print(f" - Imágenes procesadas: {len(todas_metricas_imagenes['opencv'])}\n")
    
    metricas_tecnicas.append({
        'Tecnica': 'Secuencial OpenCV',
        'Imagenes': len(todas_metricas_imagenes['opencv']),
        'Tiempo (s)': tiempo_opencv,
        'Velocidad (img/s)': velocidad_opencv,
        'Speedup': 1.0
    })
    
    # 2. Usando técnica Pool multiprocessing
    print("="*60)
    print("2. Pool (Multiprocessing)...")
    print("="*60)
    
    P = mp.cpu_count() 
    start = time.time() # tiempo inicial de la técnica Pool
    
    args_pool = []
    for filename in archivos_lista:
        nombre_base = os.path.splitext(filename)[0]
        imagen_entrada = os.path.join(folder_in, filename)
        args_pool.append((imagen_entrada, folder_out_pool, nombre_base))
    
    with mp.Pool(processes=P) as pool: # Crear un pool con P procesos
        resultados = pool.map(procesar_wrapper, args_pool)
    
    todas_metricas_imagenes['pool'] = [m for m in resultados if m is not None]
    
    end = time.time() # tiempo final de la técnica Pool
    tiempo_pool = end - start
    speedup_pool = tiempo_opencv / tiempo_pool
    velocidad_pool = len(todas_metricas_imagenes['pool']) / tiempo_pool
    
    print(f" - Tiempo: {tiempo_pool:.2f}s | Speedup: {speedup_pool:.2f}x")
    print(f" - Imágenes procesadas: {len(todas_metricas_imagenes['pool'])}\n")
    
    metricas_tecnicas.append({
        'Tecnica': f'Pool ({P} cores)',
        'Imagenes': len(todas_metricas_imagenes['pool']),
        'Tiempo (s)': tiempo_pool,
        'Velocidad (img/s)': velocidad_pool,
        'Speedup': speedup_pool
    })
    
    # 3. Usando técnica ThreadPoolExecutor   
    print("="*60)
    print("3. ThreadPoolExecutor...")
    print("="*60)
    
    num_hilos = P # En mi caso 24 hilos
    start = time.time()
    
    args_thread = []
    for filename in archivos_lista:
        nombre_base = os.path.splitext(filename)[0]
        imagen_entrada = os.path.join(folder_in, filename)
        args_thread.append((imagen_entrada, folder_out_thread, nombre_base))
    
    with ThreadPoolExecutor(max_workers=num_hilos) as executor:
        resultados = list(executor.map(procesar_wrapper, args_thread))
    
    todas_metricas_imagenes['thread'] = [m for m in resultados if m is not None]
    
    end = time.time()
    tiempo_thread = end - start
    speedup_thread = tiempo_opencv / tiempo_thread
    velocidad_thread = len(todas_metricas_imagenes['thread']) / tiempo_thread
    
    print(f" - Tiempo: {tiempo_thread:.2f}s | Speedup: {speedup_thread:.2f}x")
    print(f" - Imágenes procesadas: {len(todas_metricas_imagenes['thread'])}\n")
    
    metricas_tecnicas.append({
        'Tecnica': f'ThreadPool ({num_hilos} threads)',
        'Imagenes': len(todas_metricas_imagenes['thread']),
        'Tiempo (s)': tiempo_thread,
        'Velocidad (img/s)': velocidad_thread,
        'Speedup': speedup_thread
    })
    
    # 4. Usando técnica de librería CuPy GPU (paralelización GPU)
    print("="*60)
    print("4. CuPy GPU (Procesamiento SAR Completo)...")
    print("="*60)
    
    start = time.time()
    
    for filename in archivos_lista:
        nombre_base = os.path.splitext(filename)[0]
        imagen_entrada = os.path.join(folder_in, filename)
        metricas = procesar_imagen_sar_gpu(imagen_entrada, folder_out_cupy, nombre_base)
        if metricas:
            todas_metricas_imagenes['cupy'].append(metricas)
    
    end = time.time()
    tiempo_cupy = end - start
    speedup_cupy = tiempo_opencv / tiempo_cupy
    velocidad_cupy = len(todas_metricas_imagenes['cupy']) / tiempo_cupy
    
    print(f" - Tiempo: {tiempo_cupy:.2f}s | Speedup: {speedup_cupy:.2f}x")
    print(f" - Imágenes procesadas: {len(todas_metricas_imagenes['cupy'])}\n")
    
    metricas_tecnicas.append({
        'Tecnica': 'CuPy GPU',
        'Imagenes': len(todas_metricas_imagenes['cupy']),
        'Tiempo (s)': tiempo_cupy,
        'Velocidad (img/s)': velocidad_cupy,
        'Speedup': speedup_cupy
    })
       
    # Guardamos las métricas de técnicas
    df_tecnicas = pd.DataFrame(metricas_tecnicas)
    df_tecnicas.to_csv(os.path.join(folder_metrics, 'metricas_tecnicas_sar.csv'), index=False)
    
    # Guardar métricas de imágenes
    for tecnica, metricas_list in todas_metricas_imagenes.items():
        if metricas_list:
            df_img = pd.DataFrame(metricas_list)
            df_img.to_csv(os.path.join(folder_metrics, f'metricas_imagenes_{tecnica}.csv'), index=False)
    
    print("="*80)
    print("Resumen Comparativo de Técnicas")
    print("="*80)
    print(df_tecnicas.to_string(index=False))
    print("="*80 + "\n")
    
    