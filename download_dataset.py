import os
import gdown
import zipfile

# Ruta del dataset en Google Drive en caso de necesitarlo
# https://drive.google.com/file/d/1vNYzal2fUyaZO5zwFmWgquj6f8ghLzab/view?usp=sharing

# Configuración
file_id = "1vNYzal2fUyaZO5zwFmWgquj6f8ghLzab"
folder_out = 'dataset'
destination = os.path.join(folder_out, 'dataset.zip')
url = f'https://drive.google.com/uc?id={file_id}'

# Crear carpeta de destino
try:
    os.makedirs(folder_out, exist_ok=True)
    print(f"Directorio '{folder_out}' listo para usar")
except Exception as e:
    print(f"Error al crear el directorio: {e}")
    exit(1)

# Descargar archivo
print("\nIniciando descarga del dataset...")
try:
    gdown.download(url, destination, quiet=False)
    
    if not os.path.exists(destination):
        raise FileNotFoundError("El archivo no se descargó correctamente")
    
    file_size = os.path.getsize(destination) / (1024 * 1024)  # Tamaño en MB
    print(f"Descarga completada. Tamaño: {file_size:.2f} MB")
    
except Exception as e:
    print(f"Error durante la descarga: {e}")
    exit(1)

# Descomprimir archivo
print("\nDescomprimiendo archivo...")
try:
    if not zipfile.is_zipfile(destination):
        raise ValueError("El archivo descargado no es un ZIP válido")
    
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Extrayendo {len(file_list)} archivos/carpetas...")
        zip_ref.extractall(folder_out)
    
    print("Descompresión completada exitosamente")
    
except zipfile.BadZipFile:
    print("Error: El archivo ZIP está corrupto o dañado")
    exit(1)
except Exception as e:
    print(f"Error durante la descompresión: {e}")
    exit(1)

# Eliminar archivo ZIP
print("\nEliminando archivo ZIP...")
try:
    os.remove(destination)
    print("Archivo ZIP eliminado correctamente")
except Exception as e:
    print(f"Advertencia: No se pudo eliminar el archivo ZIP: {e}")

print("\nProceso completado con éxito")
print(f"Dataset disponible en: {os.path.abspath(folder_out)}")