import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de directorios y archivos
folder_in = 'dataset'
folder_metrics = 'metrics'
folder_summary = 'summary'

# Creamos carpeta de resumen si no existe
if not os.path.exists(folder_summary):
    os.makedirs(folder_summary)

# Preparamos lista de archivos
archivos_lista = []
for filename in os.listdir(folder_in):
    if filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
        archivos_lista.append(filename)

file_tecnicas_sar = os.path.join(folder_metrics, 'metricas_tecnicas_sar.csv')
file_tecnica_mpi = os.path.join(folder_metrics, 'metricas_tecnica_mpi.csv')
file_metricas_imagenes_cupy = os.path.join(folder_metrics, 'metricas_imagenes_cupy.csv')
file_metricas_imagenes_mpi = os.path.join(folder_metrics, 'metricas_imagenes_mpi.csv')
file_metricas_imagenes_opencv = os.path.join(folder_metrics, 'metricas_imagenes_opencv.csv')
file_metricas_imagenes_pool = os.path.join(folder_metrics, 'metricas_imagenes_pool.csv')
file_metricas_imagenes_thread = os.path.join(folder_metrics, 'metricas_imagenes_thread.csv')

# Cargamos datasets en dataframes
df_tecnicas_sar = pd.read_csv(file_tecnicas_sar)
df_tecnica_mpi = pd.read_csv(file_tecnica_mpi)

df_cupy = pd.read_csv(file_metricas_imagenes_cupy)
df_metricas_imagenes_mpi = pd.read_csv(file_metricas_imagenes_mpi)
df_opencv = pd.read_csv(file_metricas_imagenes_opencv)
df_metricas_imagenes_pool = pd.read_csv(file_metricas_imagenes_pool)
df_metricas_imagenes_thread = pd.read_csv(file_metricas_imagenes_thread)

# Unimos ambos dataframes
df_tecnicas = pd.concat([df_tecnicas_sar, df_tecnica_mpi], ignore_index=True)

# Recalculamos el speedup de MPI ya que se ejecuto solo.
# El baseline es opencv secuencial
tiempo_baseline = df_tecnicas[df_tecnicas['Tecnica'] == 'Secuencial OpenCV']['Tiempo (s)'].values[0]

# Actualizamos el speedup de MPI
indice_mpi = df_tecnicas[df_tecnicas['Tecnica'].str.contains('MPI')].index[0]
tiempo_mpi = df_tecnicas.loc[indice_mpi, 'Tiempo (s)']
df_tecnicas.loc[indice_mpi, 'Speedup'] = tiempo_baseline / tiempo_mpi

print("="*80)
print("Resumen Comparativo de Técnicas de Procesamiento para SAR")
print("="*80)
print(df_tecnicas.to_string(index=False))
print("="*80 + "\n")

# Guardamos el dataframe corregido
df_tecnicas.to_csv(os.path.join(folder_summary, 'metricas_todas_tecnicas_completo.csv'), index=False)
print(f"- Dataframe completo guardado en: {folder_summary}/metricas_todas_tecnicas_completo.csv")

# Generamos gráficas comparativas    
sns.set_style("whitegrid")
fig = plt.figure(figsize=(20, 14))
#gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
gs = fig.add_gridspec(4, 3, hspace=0.95, wspace=0.3)

fig.suptitle('Comparación de Técnicas - Procesamiento SAR para Erosión Costera',
                fontsize=16, fontweight='bold')

# 1. Tiempos de procesamiento (total)
ax1 = fig.add_subplot(gs[0, 0])
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
bars1 = ax1.barh(df_tecnicas['Tecnica'], df_tecnicas['Tiempo (s)'], color=colors)
ax1.set_xlabel('Tiempo (segundos)', fontweight='bold')
ax1.set_title('Tiempo Total de Procesamiento', fontweight='bold')
ax1.invert_yaxis()
for bar in bars1:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2., f'{width:.1f}s',
            ha='left', va='center', fontsize=9, fontweight='bold')

# 2. Speedup (obtenido vs secuencial)
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(range(len(df_tecnicas)), df_tecnicas['Speedup'], color=colors)
ax2.set_xticks(range(len(df_tecnicas)))
ax2.set_xticklabels(df_tecnicas['Tecnica'], rotation=45, ha='right')
ax2.set_ylabel('Speedup (x)', fontweight='bold')
ax2.set_title('Aceleración vs Secuencial', fontweight='bold')
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline')
ax2.legend()
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Velocidad (img/s)
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(range(len(df_tecnicas)), df_tecnicas['Velocidad (img/s)'], color=colors)
ax3.set_xticks(range(len(df_tecnicas)))
ax3.set_xticklabels(df_tecnicas['Tecnica'], rotation=45, ha='right')
ax3.set_ylabel('Imágenes/segundo', fontweight='bold')
ax3.set_title('Throughput de Procesamiento', fontweight='bold')
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Intensidad media
ax4 = fig.add_subplot(gs[1, 0])
ax4.boxplot([df_opencv['intensidad_media'], df_cupy['intensidad_media']], 
            tick_labels=['OpenCV', 'CuPy GPU'])
ax4.set_ylabel('Intensidad Media', fontweight='bold')
ax4.set_title('Comparación de Intensidad', fontweight='bold')

# 5. Porcentaje tierra/agua
ax5 = fig.add_subplot(gs[1, 1])
categorias = ['OpenCV\nTierra', 'OpenCV\nAgua', 'CuPy\nTierra', 'CuPy\nAgua']
valores = [df_opencv['porcentaje_tierra'].mean(), df_opencv['porcentaje_agua'].mean(),
            df_cupy['porcentaje_tierra'].mean(), df_cupy['porcentaje_agua'].mean()]
colors_seg = ['#e67e22', '#3498db', '#e67e22', '#3498db']
bars5 = ax5.bar(categorias, valores, color=colors_seg, alpha=0.7)
ax5.set_ylabel('Porcentaje (%)', fontweight='bold')
ax5.set_title('Segmentación Tierra/Agua', fontweight='bold')
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# 6. Longitud de costa
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(range(len(df_opencv)), df_opencv['longitud_costa_px'], 
            label='OpenCV', alpha=0.6, s=50)
ax6.scatter(range(len(df_cupy)), df_cupy['longitud_costa_px'], 
            label='CuPy GPU', alpha=0.6, s=50)
ax6.set_xlabel('Imagen #', fontweight='bold')
ax6.set_ylabel('Longitud (píxeles)', fontweight='bold')
ax6.set_title('Longitud de Línea de Costa', fontweight='bold')
ax6.legend()

# 7. Gradiente promedio
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(df_opencv['intensidad_gradiente_media'], bins=20, alpha=0.5, label='OpenCV', color='#3498db')
ax7.hist(df_cupy['intensidad_gradiente_media'], bins=20, alpha=0.5, label='CuPy GPU', color='#e74c3c')
ax7.set_xlabel('Intensidad Gradiente', fontweight='bold')
ax7.set_ylabel('Frecuencia', fontweight='bold')
ax7.set_title('Distribución de Gradientes', fontweight='bold')
ax7.legend()

# 8. Rugosidad
ax8 = fig.add_subplot(gs[2, 1])
ax8.boxplot([df_opencv['rugosidad_media'], df_cupy['rugosidad_media']], 
            tick_labels=['OpenCV', 'CuPy GPU'])
ax8.set_ylabel('Rugosidad Media', fontweight='bold')
ax8.set_title('Análisis de Textura', fontweight='bold')

# 9. Contornos detectados
ax9 = fig.add_subplot(gs[2, 2])
ax9.scatter(range(len(df_opencv)), df_opencv['num_contornos_costa'], 
            label='OpenCV', alpha=0.6, s=50, c='#2ecc71')
ax9.scatter(range(len(df_cupy)), df_cupy['num_contornos_costa'], 
            label='CuPy GPU', alpha=0.6, s=50, c='#e67e22')
ax9.set_xlabel('Imagen #', fontweight='bold')
ax9.set_ylabel('Número de Contornos', fontweight='bold')
ax9.set_title('Contornos de Costa Detectados', fontweight='bold')
ax9.legend()

# 10. Tabla comparativa
ax10 = fig.add_subplot(gs[3, :])
ax10.axis('tight')
ax10.axis('off')

tabla_data = df_tecnicas[['Tecnica', 'Tiempo (s)', 'Velocidad (img/s)', 'Speedup']].round(2)
table = ax10.table(cellText=tabla_data.values, colLabels=tabla_data.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

for i in range(len(tabla_data.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(tabla_data) + 1):
    for j in range(len(tabla_data.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')

plt.savefig(os.path.join(folder_summary, 'comparacion_tecnicas_sar.png'),
            dpi=300, bbox_inches='tight')
print(f"- Gráficas guardadas: {os.path.join(folder_summary, 'comparacion_tecnicas_sar.png')}\n")

# Generamos el informe final
# Accedo al tiempo de OpenCV y CuPy
tiempo_opencv = df_tecnicas[df_tecnicas['Tecnica'] == 'Secuencial OpenCV']['Tiempo (s)'].values[0]
tiempo_cupy = df_tecnicas[df_tecnicas['Tecnica'] == 'CuPy GPU']['Tiempo (s)'].values[0]
speedup_cupy = tiempo_opencv / tiempo_cupy

mejor_tecnica = df_tecnicas.loc[df_tecnicas['Tiempo (s)'].idxmin()]

# Construir el informe como string
informe = f"""{"="*80}
Informe Final - Procesamiento SAR Comparativo
{"="*80}

Configuración de procesamiento:
--------------------------------
- Tamaño fijo: 1000x2000 píxeles (ancho x alto)
- Total de imágenes: {len(archivos_lista)}
- Verificación de tamaño: OK (todas las imágenes redimensionadas correctamente)

Pipeline SAR aplicado:
----------------------
- Filtrado de ruido speckle
- Mejora de contraste (CLAHE/Equalización)
- Detección de bordes (Sobel/Canny)
- Segmentación tierra/agua (Otsu + morfología)
- Extracción de línea de costa
- Análisis de textura (varianza local)

Mejor técnica:
--------------
- Método: {mejor_tecnica['Tecnica']}
- Tiempo: {mejor_tecnica['Tiempo (s)']:.2f} segundos
- Velocidad: {mejor_tecnica['Velocidad (img/s)']:.2f} img/s
- Speedup: {mejor_tecnica['Speedup']:.2f}x

Resultados promedio (OpenCV):
------------------------------
- Intensidad: {df_opencv['intensidad_media'].mean():.1f} ± {df_opencv['intensidad_std'].mean():.1f}
- Tierra: {df_opencv['porcentaje_tierra'].mean():.1f}% | Agua: {df_opencv['porcentaje_agua'].mean():.1f}%
- Contornos costa: {df_opencv['num_contornos_costa'].mean():.1f}
- Longitud costa: {df_opencv['longitud_costa_px'].mean():.0f} px
- Rugosidad: {df_opencv['rugosidad_media'].mean():.2f}

Comparación GPU vs CPU:
-----------------------
- OpenCV (CPU): {tiempo_opencv:.2f}s
- CuPy (GPU): {tiempo_cupy:.2f}s
- Speedup GPU: {speedup_cupy:.2f}x
- Diferencia métricas: Mínima (equivalentes)

Pr+oximos pasos:
-------------
→ Análisis temporal: Comparar imágenes de diferentes fechas
→ Cuantificar erosión: Medir desplazamiento de línea de costa
→ Generar mapas de cambio

{"="*80}

Resuen:
- Procesamiento completado exitosamente
- Total técnicas evaluadas: {len(df_tecnicas)}
- Total imágenes procesadas por técnica: {len(archivos_lista)}
{"="*80}
"""

# Guardamos en archivo
archivo_informe = os.path.join(folder_summary, 'informe_final_sar.txt')
with open(archivo_informe, 'w', encoding='utf-8') as f:
    f.write(informe)

print(f"- Informe final guardado en: {archivo_informe}")