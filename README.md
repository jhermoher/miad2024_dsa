## Repositorio del Proyecto Final
# CATEGORIZACIÓN Y ANÁLISIS DE SENTIMIENTOS DE ARTÍCULOS DE NOTICIAS
## DESPLIEGUE DE SOLUCIONES ANALÍTICAS
### Curso miid4304-202415


### RESUMEN
La sobrecarga de información en la actualidad presenta desafíos tanto para los lectores como para analistas y tomadores de decisiones. Uno de estos desafíos, reconocido como una problemática social, es la polarización de la opinión pública, impulsada por la difusión de noticias de diversas categorías a través de artículos en línea.

La validación y refinamiento de la categorización de artículos de noticias es crucial para comprender su contenido. Además, la identificación del sentimiento general (positivo, neutro o negativo) en los diversos temas de noticias puede ser utilizada por creadores de contenido, medios de comunicación e incluso legisladores para lanzar campañas o políticas que controlen su difusión con el propósito de mitigar su impacto.


### TABLA DE CONTENIDO - README
- [DATOS](#datos)
- [METODOLOGÍA](#metodología)
- [RESULTADOS](#resultados)


### DATOS
Los datos corresponden a artículos de noticias del portal HuffPost en Estados Unidos que fueron recopilados entre 2012 y 2022. El dataset cuenta con cerca de 210 mil titulares de noticias.

La fuente de los datos usada en este proyecto se encuentra disponible en **https://www.kaggle.com/datasets/rmisra/news-category-datase**.

Créditos:
- Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv: 2209.11429 (2022).
- Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 9798585463570 (2021).

Para más detalles, refiérase al archivo ```data/README.md```


### METODOLOGÍA
Este proyecto tiene como objetivo desarrollar un sistema automatizado para la validación, refinamiento y posible re-categorización de artículos de noticias, así como el análisis de sentimientos. El sistema empleará un enfoque híbrido de aprendizaje de máquina supervisado y no supervisado para la categorización, y utilizará VADER para el análisis de sentimientos, con el fin de comprender y potencialmente mitigar la polarización de la opinión pública.

La herramienta está montada en una instancia EC2 de los servicios ofrecidos por AWS.

Para más detalles, refiérase al archivo ```notebooks/methodology.md```.

Los reportes elaborados durante el desarrollo de este proyecto están disponibles en la carpeta ```submittals/```.


### RESULTADOS
```[Por definir]´´´