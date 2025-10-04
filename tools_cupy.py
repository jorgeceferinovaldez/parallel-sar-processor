import os
# Para evitar errores de OpenCV con imágenes grandes (> 2^24 píxeles)
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**40)

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
from cupyx.scipy.ndimage import zoom


def procesar_imagen_sar_gpu(ruta_entrada, carpeta_salida, nombre_base, tamanio_fijo=(1000, 2000)):
    """
    Procesa imágenes SAR en GPU usando CuPy para procesamiento de imágenes acelerado.
    Esta función realiza un pipeline completo de procesamiento de imágenes SAR enteramente en GPU,
    incluyendo redimensionamiento, filtrado, mejora, detección de bordes, segmentación y análisis
    de textura. Los resultados se transfieren de vuelta a CPU para detección de línea de costa y
    cálculo de métricas.
    
    Args:
        ruta_entrada (str): Ruta al archivo de imagen SAR de entrada
        carpeta_salida (str): Directorio de salida para imágenes procesadas
        nombre_base (str): Nombre base para archivos de salida y métricas
        tamanio_fijo (tuple, optional): Tamaño objetivo de imagen (ancho, alto). Por defecto (1000, 2000)
    
    Returns:
        dict or None: Diccionario conteniendo métricas de procesamiento incluyendo:
            - nombre: Nombre base
            - intensidad_media: Intensidad promedio
            - intensidad_std: Desviación estándar de intensidad
            - intensidad_gradiente_media: Magnitud de gradiente promedio
            - porcentaje_tierra: Porcentaje de área terrestre
            - porcentaje_agua: Porcentaje de área acuática
            - num_contornos_costa: Número de contornos de costa
            - longitud_costa_px: Longitud de costa en píxeles
            - rugosidad_media: Rugosidad de textura promedio
        Retorna None si falla la carga de imagen o ocurre error de procesamiento
    
    Pipeline de Procesamiento:
        1. Carga y redimensionamiento de imagen a tamaño fijo
        2. Aplicación de filtro Gaussiano para reducción de ruido
        3. Ecualización de histograma para mejora de contraste
        4. Cálculo de detección de bordes Sobel
        5. Aplicación de umbralización Otsu para segmentación
        6. Operaciones morfológicas (erosión + dilatación)
        7. Cálculo de varianza local para análisis de textura
        8. Detección de contornos de costa y cálculo de métricas
    
    Archivos de Salida:
        - enhanced.png: Imagen con contraste mejorado
        - edges.png: Resultado de detección de bordes
        - segmentation.png: Máscara binaria de segmentación
        - texture.png: Mapa de varianza de textura
        - coastline.png: Superposición de línea de costa detectada
    """

    try:
        # 1. Carga y redimensionamiento
        img_cpu = cv2.imread(ruta_entrada, cv2.IMREAD_GRAYSCALE)
        if img_cpu is None:
            return None
        
        # Transferimos a GPU
        img_gpu = cp.asarray(img_cpu, dtype=cp.float32)
        
        # Redimensionamos con zoom
        factor_h = tamanio_fijo[1] / img_gpu.shape[0]
        factor_w = tamanio_fijo[0] / img_gpu.shape[1]
        img_gpu = zoom(img_gpu, (factor_h, factor_w), order=1)
        
        # Verificamos tamaño final
        alto_final, ancho_final = img_gpu.shape
        assert (ancho_final, alto_final) == tamanio_fijo, \
            f"Error GPU: tamaño esperado {tamanio_fijo}, obtenido ({ancho_final}, {alto_final})"
        
        # Normalizamos
        img_gpu = (img_gpu - cp.min(img_gpu)) / (cp.max(img_gpu) - cp.min(img_gpu) + 1e-8) * 255.0
        
        # 2. Filtrado Gaussiano (aproximación al bilateral en GPU)
        from cupyx.scipy.ndimage import gaussian_filter
        img_denoised = gaussian_filter(img_gpu, sigma=2.0)
        
        # 3. Mejora de contraste (Histogram Equalization en GPU)
        # Convertimos a uint8 para histogram
        img_uint8 = cp.clip(img_denoised, 0, 255).astype(cp.uint8)
        
        # Calculamos histograma en GPU
        hist = cp.histogram(img_uint8, bins=256, range=(0, 256))[0]
        cdf = cp.cumsum(hist)
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype(cp.uint8)
        
        # Aplicamos ecualización
        img_enhanced = cdf_normalized[img_uint8]
        img_enhanced = img_enhanced.astype(cp.float32)
        
        # 4. Detección de bordes (Sobel en GPU)
        from cupyx.scipy.ndimage import sobel
        sobelx = sobel(img_enhanced, axis=1)
        sobely = sobel(img_enhanced, axis=0)
        gradient_mag = cp.sqrt(sobelx**2 + sobely**2)
        gradient_mag = (gradient_mag - cp.min(gradient_mag)) / (cp.max(gradient_mag) - cp.min(gradient_mag) + 1e-8) * 255.0
        
        # 5. Segmentación (Otsu en GPU)
        img_uint8 = img_enhanced.astype(cp.uint8)
        hist = cp.histogram(img_uint8, bins=256, range=(0, 256))[0].astype(cp.float32)
        hist = hist / cp.sum(hist)
        
        # Algoritmo de Otsu en GPU
        bins = cp.arange(256)
        max_variance = 0
        threshold = 0
        
        for t in range(1, 255):
            w0 = cp.sum(hist[:t])
            w1 = cp.sum(hist[t:])
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = cp.sum(bins[:t] * hist[:t]) / w0
            mu1 = cp.sum(bins[t:] * hist[t:]) / w1
            
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                threshold = t
        
        binary_otsu = (img_enhanced > threshold).astype(cp.uint8) * 255
        
        # 6. Morfología (erosión/dilatación en GPU)
        from cupyx.scipy.ndimage import binary_erosion, binary_dilation
        kernel = cp.ones((5, 5), dtype=bool)
        binary_cleaned = binary_dilation(binary_erosion(binary_otsu > 0, kernel), kernel).astype(cp.uint8) * 255
        
        # 7. Análisis de textura (Varianza local en GPU)
        from cupyx.scipy.ndimage import uniform_filter
        kernel_size = 15
        img_mean = uniform_filter(img_enhanced, size=kernel_size)
        img_sqr_mean = uniform_filter(img_enhanced**2, size=kernel_size)
        img_variance = img_sqr_mean - img_mean**2
        img_variance = (img_variance - cp.min(img_variance)) / (cp.max(img_variance) - cp.min(img_variance) + 1e-8) * 255.0
        
        # 8. Transferimos resultados a CPU y guardamos
        img_enhanced_cpu = cp.asnumpy(img_enhanced).astype(np.uint8)
        gradient_mag_cpu = cp.asnumpy(gradient_mag).astype(np.uint8)
        binary_cleaned_cpu = cp.asnumpy(binary_cleaned).astype(np.uint8)
        img_variance_cpu = cp.asnumpy(img_variance).astype(np.uint8)
        
        carpeta_imagen = os.path.join(carpeta_salida, nombre_base)
        os.makedirs(carpeta_imagen, exist_ok=True)
        
        cv2.imwrite(os.path.join(carpeta_imagen, 'enhanced.png'), img_enhanced_cpu)
        cv2.imwrite(os.path.join(carpeta_imagen, 'edges.png'), gradient_mag_cpu)
        cv2.imwrite(os.path.join(carpeta_imagen, 'segmentation.png'), binary_cleaned_cpu)
        cv2.imwrite(os.path.join(carpeta_imagen, 'texture.png'), img_variance_cpu)
        
        # 9. Calculamos métricas y generamos imagen de línea de costa en CPU
        # Detección de línea de costa en CPU
        contours, _ = cv2.findContours(binary_cleaned_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coastline_length = 0
        num_contours = 0
        
        # Creamos imagen de línea de costa
        coastline_cpu = np.zeros_like(binary_cleaned_cpu)
        
        if len(contours) > 0:
            contours_filtered = [c for c in contours if cv2.contourArea(c) > 1000]
            cv2.drawContours(coastline_cpu, contours_filtered, -1, 255, 2)
            coastline_length = sum([cv2.arcLength(c, True) for c in contours_filtered])
            num_contours = len(contours_filtered)
        
        # Guardamos imagen de línea de costa
        cv2.imwrite(os.path.join(carpeta_imagen, 'coastline.png'), coastline_cpu)
        
        metricas = {
            'nombre': nombre_base,
            'intensidad_media': float(np.mean(img_enhanced_cpu)),
            'intensidad_std': float(np.std(img_enhanced_cpu)),
            'num_bordes_canny': 0,  # No calculamos Canny en GPU
            'intensidad_gradiente_media': float(np.mean(gradient_mag_cpu)),
            'porcentaje_tierra': float(np.sum(binary_cleaned_cpu > 0) / binary_cleaned_cpu.size * 100),
            'porcentaje_agua': float(np.sum(binary_cleaned_cpu == 0) / binary_cleaned_cpu.size * 100),
            'num_contornos_costa': num_contours,
            'longitud_costa_px': float(coastline_length),
            'rugosidad_media': float(np.mean(img_variance_cpu))
        }
        
        # Liberamos memoria GPU
        del img_gpu, img_denoised, img_enhanced, gradient_mag, binary_otsu, binary_cleaned, img_variance
        cp.cuda.Stream.null.synchronize()
        
        return metricas
        
    except Exception as e:
        print(f"Error procesando GPU {ruta_entrada}: {str(e)}")
        return None
    
