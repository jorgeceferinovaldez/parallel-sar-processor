import os
# Para evitar errores de OpenCV con imágenes grandes (> 2^24 píxeles)
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**40) 

import time
import numpy as np
from mpi4py import MPI
import pandas as pd

from tools_img import procesar_imagen_sar_costera

if __name__ == '__main__':
    folder_in = 'dataset'
    folder_out = 'output'
    folder_metrics = 'metrics'

    folder_out_mpi = os.path.join(folder_out, 'sar_mpi')
    os.makedirs(folder_metrics, exist_ok=True)
    
    # Procesamiento MPI 
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0: # Proceso maestro
        print("\n" + "="*80)
        print("Procesamiento MPI - SAR para detección de erosión costera")
        print("="*80)
        
        os.makedirs(folder_out, exist_ok=True)
        os.makedirs(folder_out_mpi, exist_ok=True)
        
        # Preparamos la lista de archivos de imágenes
        archivos = []
        for filename in os.listdir(folder_in):
            nombre_base = os.path.splitext(filename)[0]
            imagen_entrada = os.path.join(folder_in, filename)
            archivos.append((imagen_entrada, folder_out_mpi, nombre_base))
        
        print(f"Total de archivos: {len(archivos)}")
        print(f"Numero de procesos MPI: {size}")
        print("Tamaño fijo de procesamiento: 1000x2000 píxeles (ancho x alto)")
        print("\nPipeline de procesamiento:")
        print("  1. Filtrado de ruido speckle (SAR)")
        print("  2. Mejora de contraste (CLAHE)")
        print("  3. Detección de bordes (Sobel)")
        print("  4. Segmentación tierra/agua (Otsu)")
        print("  5. Extracción de línea de costa")
        print("  6. Análisis de textura (Varianza)")
        print("-"*80)
        
        start_mpi = time.time() # Tiempo de inicio MPI
    else: # Otros procesos
        archivos = None
        start_mpi = None
    
    # Broadcast de datos a todos los procesos
    archivos = comm.bcast(archivos, root=0)
    start_mpi = comm.bcast(start_mpi, root=0)
    
    # Distribución cíclica de archivos entre procesos
    mis_archivos = archivos[rank::size]
    
    # Procesar archivos asignados
    inicio_proceso = time.time()
    procesados = 0
    errores = 0
    metricas_locales = []
    
    for entrada, salida, nombre in mis_archivos:
        metricas = procesar_imagen_sar_costera(entrada, salida, nombre, tamanio_fijo=(1000, 2000))
        if metricas is not None:
            procesados += 1
            metricas_locales.append(metricas)
        else:
            errores += 1
    
    fin_proceso = time.time()
    tiempo_proceso = fin_proceso - inicio_proceso
    
    # Recolectamos estadísticas
    total_procesados = comm.reduce(procesados, op=MPI.SUM, root=0)
    total_errores = comm.reduce(errores, op=MPI.SUM, root=0)
    tiempos_procesos = comm.gather(tiempo_proceso, root=0)
    imagenes_por_proceso = comm.gather(procesados, root=0)
    
    # Recolectamos métricas de todas las imágenes
    todas_metricas = comm.gather(metricas_locales, root=0)
    
    comm.Barrier() # Sincronización de procesos
    
    # Análisis y visualización
    if rank == 0: # Proceso maestro
        end_mpi = time.time() # Tiempo de fin MPI
        tiempo_total = end_mpi - start_mpi
        velocidad = total_procesados / tiempo_total if tiempo_total > 0 else 0
        
        print("-"*80)
        print(f"\n Procesamiento completado")
        print(f"  - Imágenes procesadas: {total_procesados}")
        print(f"  - Errores: {total_errores}")
        print(f"  - Tiempo total: {tiempo_total:.2f} segundos")
        print(f"  - Velocidad: {velocidad:.2f} imágenes/segundo")
        print("="*80 + "\n")
        
        # Combinamos todas las métricas
        metricas_completas = []
        for metricas_proceso in todas_metricas:
            metricas_completas.extend(metricas_proceso)
        
        # Creamois DataFrames
        df_imagenes = pd.DataFrame(metricas_completas)
        
        # Guardamos las métricas de la técnica MPI en un dataframe
        metricas_tecnica = {
            'Tecnica': f'MPI ({size} cores)',
            'Imagenes': total_procesados,
            'Tiempo (s)': tiempo_total,
            'Velocidad (img/s)': velocidad,
            'Speedup': 1.0  # Se calculará después al comparar con otras técnicas
        }
        
        df_tecnica = pd.DataFrame([metricas_tecnica])
        
        # Calculamos las métricas por proceso
        df_procesos = pd.DataFrame({
            'Proceso': range(size),
            'Imagenes Procesadas': imagenes_por_proceso,
            'Tiempo (s)': tiempos_procesos,
            'Velocidad (img/s)': [img/t if t > 0 else 0 for img, t in zip(imagenes_por_proceso, tiempos_procesos)]
        })
        
        # Guardamos los archivos CSVs 
        df_tecnica.to_csv(os.path.join(folder_metrics, 'metricas_tecnica_mpi.csv'), index=False)
        df_imagenes.to_csv(os.path.join(folder_metrics, 'metricas_imagenes_mpi.csv'), index=False)
        df_procesos.to_csv(os.path.join(folder_metrics, 'metricas_procesos_mpi.csv'), index=False)

        print("="*80)
        print("Resumen Estadístico - Técnica MPI")
        print("="*80)
        print(df_tecnica.to_string(index=False))
        print("="*80 + "\n")
        
        # Estadissticas de las imágenes
        print("="*80)
        print("Estadísticas de imágenes SAR procesadas")
        print("="*80)
        print(f"\nNúmero total de imágenes: {len(df_imagenes)}")
        print(f"\nIntensidad promedio: {df_imagenes['intensidad_media'].mean():.2f} ± {df_imagenes['intensidad_std'].mean():.2f}")
        print("\nSegmentación tierra/agua:")
        print(f"  - Tierra promedio: {df_imagenes['porcentaje_tierra'].mean():.2f}%")
        print(f"  - Agua promedio: {df_imagenes['porcentaje_agua'].mean():.2f}%")
        print("\nLínea de costa:")
        print(f"  - Contornos detectados (promedio): {df_imagenes['num_contornos_costa'].mean():.1f}")
        print(f"  - Longitud costa (promedio): {df_imagenes['longitud_costa_px'].mean():.1f} px")
        print("\nTextura:")
        print(f"  - Rugosidad media: {df_imagenes['rugosidad_media'].mean():.2f}")
        print("\nRendimiento MPI:")
        print(f"  - Eficiencia: {(np.mean(tiempos_procesos) / max(tiempos_procesos) * 100):.1f}%")
        print(f"  - Balance de carga: {(min(imagenes_por_proceso) / max(imagenes_por_proceso) * 100):.1f}%")
        print("="*80 + "\n")
        
        