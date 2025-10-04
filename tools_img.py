import os
# Para evitar errores de OpenCV con imágenes grandes (> 2^24 píxeles)
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**40)

import cv2
import numpy as np


def procesar_imagen_sar_costera(ruta_entrada, carpeta_salida, nombre_base, tamanio_fijo=(1000, 2000)):
    """
    Procesa imágenes SAR costeras mediante un pipeline de múltiples etapas para análisis de línea de costa.
    
    Args:
        ruta_entrada (str): Ruta al archivo de imagen SAR de entrada
        carpeta_salida (str): Directorio de salida donde se guardarán las imágenes procesadas
        nombre_base (str): Nombre base usado para crear subdirectorio de salida y nombrar archivos
        tamanio_fijo (tuple, opcional): Tamaño objetivo de la imagen (ancho, alto). Por defecto (1000, 2000)
    
    Returns:
        dict or None: Diccionario conteniendo métricas de procesamiento incluyendo:
            - nombre: Identificador del nombre base
            - intensidad_media: Intensidad promedio de la imagen mejorada
            - intensidad_std: Desviación estándar de la intensidad
            - num_bordes_canny: Número de píxeles de bordes detectados con Canny
            - intensidad_gradiente_media: Intensidad promedio del gradiente
            - porcentaje_tierra: Porcentaje de píxeles de tierra en la segmentación
            - porcentaje_agua: Porcentaje de píxeles de agua en la segmentación
            - num_contornos_costa: Número de contornos de línea de costa detectados
            - longitud_costa_px: Longitud total de la línea de costa en píxeles
            - rugosidad_media: Rugosidad/variación de textura promedio
        
        Retorna None si el procesamiento falla o la imagen de entrada no puede cargarse.
    
    Pipeline de Procesamiento:
        1. Carga de imagen y redimensionamiento a dimensiones fijas
        2. Reducción de ruido speckle usando filtro bilateral y desenfoque de medias no locales
        3. Mejora de contraste usando CLAHE
        4. Detección de bordes usando operadores Canny y Sobel
        5. Segmentación tierra/agua usando umbralización Otsu y operaciones morfológicas
        6. Extracción de línea de costa mediante detección de contornos
        7. Análisis de textura usando filtrado de varianza
        8. Salida de 5 imágenes procesadas: mejorada, bordes, segmentación, línea de costa y textura
    
    Raises:
        AssertionError: Si las dimensiones de la imagen redimensionada no coinciden con el tamaño esperado
        Varias excepciones de OpenCV/NumPy: Si fallan las operaciones de procesamiento de imagen
    """

    try:
        img = cv2.imread(ruta_entrada, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        img = cv2.resize(img, tamanio_fijo, interpolation=cv2.INTER_AREA)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        alto_final, ancho_final = img.shape
        assert (ancho_final, alto_final) == tamanio_fijo, \
            f"Error: tamaño esperado {tamanio_fijo}, obtenido ({ancho_final}, {alto_final})"
        
        # 2. Filtrado de ruido speckle
        img_denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        img_denoised = cv2.fastNlMeansDenoising(img_denoised, h=10, 
                                                templateWindowSize=7, 
                                                searchWindowSize=21)
        
        # 3. Mejora de contraste con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_denoised)
        
        # 4. Detección de bordes
        edges_canny = cv2.Canny(img_enhanced, threshold1=50, threshold2=150)
        sobelx = cv2.Sobel(img_enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 5. Segmentamos tierra y agua
        _, binary_otsu = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_cleaned = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel_morph)
        binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel_morph)
        
        # 6. Extraemos la línea de costa
        contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coastline = np.zeros_like(img)
        coastline_length = 0
        num_contours = 0
        
        if len(contours) > 0:
            contours_filtered = [c for c in contours if cv2.contourArea(c) > 1000]
            cv2.drawContours(coastline, contours_filtered, -1, 255, 2)
            coastline_length = sum([cv2.arcLength(c, True) for c in contours_filtered])
            num_contours = len(contours_filtered)
        
        # 7. Análisis de textura
        kernel_texture = np.ones((15, 15), np.float32) / 225
        img_mean = cv2.filter2D(img_enhanced.astype(np.float32), -1, kernel_texture)
        img_sqr_mean = cv2.filter2D((img_enhanced.astype(np.float32))**2, -1, kernel_texture)
        img_variance = img_sqr_mean - img_mean**2
        img_variance = cv2.normalize(img_variance, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 8. Guardamos solo las 5 imágenes principales
        carpeta_imagen = os.path.join(carpeta_salida, nombre_base)
        os.makedirs(carpeta_imagen, exist_ok=True)
        
        resultados = {
            'enhanced': img_enhanced,
            'edges': gradient_mag,
            'segmentation': binary_cleaned,
            'coastline': coastline,
            'texture': img_variance
        }
        
        for nombre, imagen in resultados.items():
            cv2.imwrite(os.path.join(carpeta_imagen, f"{nombre}.png"), imagen)
        
        # 9. Calculamos métricas
        metricas = {
            'nombre': nombre_base,
            'intensidad_media': float(np.mean(img_enhanced)),
            'intensidad_std': float(np.std(img_enhanced)),
            'num_bordes_canny': int(np.sum(edges_canny > 0)),
            'intensidad_gradiente_media': float(np.mean(gradient_mag)),
            'porcentaje_tierra': float(np.sum(binary_cleaned > 0) / binary_cleaned.size * 100),
            'porcentaje_agua': float(np.sum(binary_cleaned == 0) / binary_cleaned.size * 100),
            'num_contornos_costa': num_contours,
            'longitud_costa_px': float(coastline_length),
            'rugosidad_media': float(np.mean(img_variance))
        }
        
        return metricas
        
    except Exception as e:
        print(f"Error procesando {ruta_entrada}: {str(e)}")
        return None

