# CATEGORIZACIÓN Y ANÁLISIS DE SENTIMIENTOS DE ARTÍCULOS DE NOTICIAS 🗞️
## PROYECTO FINAL - DESPLIEGUE DE SOLUCIONES ANALÍTICAS
### Curso miid4304-202415

## 📋 Tabla de Contenido
- [Resumen](#resumen)
- [Problemática](#problemática)
- [Datos](#datos)
- [Metodología](#metodología)
  - [Categorización y Re-categorización](#categorización-y-re-categorización)
  - [Análisis de Sentimientos](#análisis-de-sentimientos)
  - [Panel de Control](#panel-de-control)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación y Uso](#instalación-y-uso)
- [Resultados y Visualizaciones](#resultados-y-visualizaciones)
- [Créditos y Referencias](#créditos-y-referencias)

## 📝 Resumen
La sobrecarga de información en la actualidad presenta desafíos significativos tanto para lectores como para analistas y tomadores de decisiones. Este proyecto aborda específicamente la problemática de la polarización de la opinión pública, impulsada por la difusión de noticias a través de artículos en línea, mediante el desarrollo de herramientas analíticas para la categorización y análisis de sentimientos de artículos de noticias.

## ⚠️ Problemática
La polarización de la opinión pública a través de la difusión de noticias en línea representa un desafío significativo para la sociedad actual. La validación y refinamiento de la categorización de artículos de noticias, junto con la identificación de su sentimiento (positivo, neutro o negativo), son cruciales para:
- Comprender patrones en la difusión de noticias
- Identificar tendencias en la polarización de contenidos
- Proporcionar insights para la toma de decisiones informada
- Facilitar el desarrollo de estrategias para mitigar impactos negativos

## 📊 Datos
- **Fuente**: Artículos de noticias del portal HuffPost (Estados Unidos)
- **Período**: 2012-2022
- **Tamaño**: Aproximadamente 210,000 titulares de noticias
- **Disponibilidad**: [News Category Dataset (Kaggle)](https://www.kaggle.com/datasets/rmisra/news-category-dataset)

Para más detalles sobre la estructura y contenido de los datos, consulte el archivo `data/README.md`.

## 🔍 Metodología
El proyecto implementa un enfoque híbrido innovador que combina:

### Categorización y Re-categorización
1. **Análisis No Supervisado**:
   - Implementación de técnicas de modelado de temas (LDA/NMF)
   - Descubrimiento de patrones temáticos latentes
   - Identificación de subcategorías emergentes

2. **Análisis Supervisado**:
   - Utilización de etiquetas existentes para entrenamiento
   - Evaluación de precisión contra categorías predefinidas
   - Refinamiento del modelo de clasificación

3. **Integración y Refinamiento**:
   - Comparación de resultados supervisados y no supervisados
   - Identificación de discrepancias
   - Refinamiento de estructura de categorías

### Análisis de Sentimientos
- Implementación de VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Análisis de sentimientos positivos, negativos y neutrales
- Agregación de resultados por categoría y evolución temporal

### Panel de Control
Visualización interactiva de:
- Distribución y evolución de categorías
- Tendencias de sentimiento
- Indicadores de polarización
- Análisis de frecuencia de palabras

## 📁 Estructura del Repositorio
```
proyecto/
│
├── data/              # Datos y documentación relacionada
├── notebooks/         # Jupyter notebooks con análisis
├── src/               # Código fuente del proyecto
├── dashboard/         # Código del panel de control
├── docs/              # Documentación adicional
├── tests/             # Pruebas unitarias y de integración
├── submittals/        # Reportes y entregables
└── requirements.txt   # Dependencias del proyecto
```

## 🚀 Instalación y Uso
1. Clone el repositorio:
```bash
git clone https://github.com/jhermoher/miad2024_dsa_G16.git
```

2. Instale las dependencias:
```bash
pip install -r requirements.txt
```

3. Siga las instrucciones en `docs/setup.md` para la configuración del entorno.

## 📈 Resultados y Visualizaciones
El panel de control interactivo está disponible en la instancia EC2 de AWS:
- URL: [Por definir]
- Credenciales: [Por definir]

Para acceder a reportes detallados y análisis, consulte la carpeta `submittals/`.

## 🏆 Créditos y Referencias
- Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv: 2209.11429 (2022).
- Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).

### Mantenedores
- Andrés Gualdrón - a.gualdrong@uniandes.edu.co
- Jersson Morales - j.moralesh@uniandes.edu.co
- Juan Manzano - j.manzano@uniandes.edu.co
- Lizebeth Ordoñez - cl.ordoneza@uniandes.edu.co

### Licencia
[Por definir]
