from matplotlib import pyplot as plt 
import seaborn as sns
from pathlib import Path
import pandas as pd

data = pd.read_csv(Path(Path.home(), 'data', 'MPLS', 'all_data.csv'))
# Distribución de la edad
plt.figure(figsize=(8, 5))
sns.histplot(data['Edad'].dropna(), kde=True, bins=15)
plt.title('Distribución de la Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(Path(Path.home(), 'data', 'MPLS', 'age_distribution.png'))

# Distribución por género
plt.figure(figsize=(6, 5))
sns.countplot(y='Hombre/ Mujer', data=data, palette='muted')
plt.title('Distribución de Género')
plt.xlabel('Frecuencia')
plt.ylabel('Género')
plt.tight_layout()
plt.savefig(Path(Path.home(), 'data', 'MPLS','sex_distribution.png'))

# Participación en microconciertos
microconcert_columns = ['Primer microconcierto', 'Segundo microconcierto', 'Tercer Microconcierto']
participation_counts = data[microconcert_columns].notnull().sum()

plt.figure(figsize=(8, 5))
participation_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Participación en Microconciertos')
plt.ylabel('Cantidad de Participantes')
plt.xlabel('Microconciertos')
plt.tight_layout()
plt.savefig(Path(Path.home(), 'data', 'MPLS','microconcert_participation.png'))

# 1. Conteo de tareas completadas por participante
# Identificar las columnas relacionadas con tareas basadas en el formato descrito
tasks_columns = [col for col in data.columns if '__' in col]

# Extraer los nombres únicos de las tareas
tasks = set([col.split('_')[0] for col in tasks_columns])

# Crear un diccionario para almacenar el número de personas que completaron cada tarea
task_completion = {}

for task in tasks:
    # Seleccionar todas las columnas relacionadas con esta tarea
    task_related_columns = [col for col in tasks_columns if col.startswith(task)]
    
    # Contar cuántas personas tienen al menos un valor no nulo en estas columnas
    task_completion[task] = data[task_related_columns].notnull().any(axis=1).sum()

# Convertir los resultados en un DataFrame para mayor claridad
task_completion_df = pd.DataFrame(list(task_completion.items()), columns=['Tarea', 'Personas que completaron'])

# Ordenar por cantidad de personas que completaron
task_completion_df = task_completion_df.sort_values(by='Personas que completaron', ascending=False)

print("Conteo de tareas completadas por participante:")
print(task_completion_df)

task_completion_df.to_csv(Path(Path.home(), 'data', 'MPLS','task_completion.csv'), index=False)

# 2. Distribución de valores del Minimental
minimental_distribution = data['Minimental'].dropna()

# Participantes con Minimental mayor a 30
minimental_above_30 = len(data[data['Minimental'] > 30])
print(f"Cantidad de participantes con Minimental mayor a 30: {minimental_above_30}")

# Generar gráfico de la distribución del Minimental
plt.figure(figsize=(8, 5))
sns.histplot(minimental_distribution, kde=True, bins=15, color='purple')
plt.title('Distribución de valores del Minimental')
plt.xlabel('Puntaje Minimental')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(Path(Path.home(), 'data', 'MPLS','minimental_distribution.png'))

# 3. Cantidad de participantes por idioma
if 'language' in data.columns:
    language_counts = data['language'].value_counts()
    print("Cantidad de participantes por idioma:")
    print(language_counts)

    # Generar gráfico de la cantidad de participantes por idioma
    plt.figure(figsize=(10, 6))
    language_counts.plot(kind='bar', color='teal', edgecolor='black')
    plt.title('Cantidad de participantes por idioma')
    plt.ylabel('Cantidad de participantes')
    plt.xlabel('Idioma')
    plt.tight_layout()
    plt.savefig(Path(Path.home(), 'data', 'MPLS','language_distribution.png'))
else:
    print("La columna 'language' no está disponible en el dataset.")