# Imágenes por microondas

En estos jupyter notebooks describiremos algunos de los métodos utilizados en la reconstrucción de imágenes obtenidas a partir de microondas.

### Requerimientos

- Tener Python 3.5 o más nuevo instalado. La versión de Python puede verificarse con `python3 --version` en la línea de comandos. La última versión de Python puede descargarse de [aquí](https://www.python.org/downloads/).
- Tener [instalado Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html).
    - Nota: En sistemas derivados de Debian la instalación es simplemente `apt install jupyter`


### Utilización
- Clonar o descargar este repositorio.
- Ejecutar `jupyter notebook` en la línea de comando dentro del directorio del repositorio.
- Se abrirá una sesión de notebook de Jupyter en el navegador y se puede comenzar a navegar a través de los notebooks disponibles.

---

## Temas

1. [Problema directo](https://github.com/rirastorza/Intro2MI/tree/main/problema_directo) aquí se comentarán tres métodos numéricos para resolver el problema directo: FDTD, FEM, y MoM.<br>
   a. [FDTD](https://github.com/rirastorza/Intro2MI/blob/main/problema_directo/sim2.ipynb). Se comenta el uso de las herramientas de simulación con diferencias finitas con un ejemplo de calibración del setup experimental.<br>
   b. [FEM](https://github.com/rirastorza/Intro2MI/blob/main/problema_directo/script_Einc_fem.py). Se comenta la simulación del problema directo utilizando el método de elmentos finitos. En particular se compara la solución con la analítica.<br>
   c. [MoM](https://github.com/rirastorza/Intro2MI/blob/main/problema_directo/ejemplo_MoM_FDTD.ipynb). Se comenta la simulación del problema directo utilizando el método de los momentos. En particular se compara la solución con la analítica.<br>
   d. [Simulación de fantomas](https://github.com/rirastorza/Intro2MI/blob/main/problema_directo/ejemplo_contrastes.ipynb). Se comenta la simulación de fantomas que se utililzarán en la etapa de calibración.<br>
   e. [Calibración](https://github.com/rirastorza/Intro2MI/blob/main/problema_directo/sim0.ipynb). Se comenta la simulación de un cilindro centrado y descentrado con FDTD y la comparación con datos experimentales.<br>

2. [Método deterministicos:](https://github.com/rirastorza/Intro2MI/tree/main/metodos_deterministicos) aquí se comentarán los métodos no iterativos: Born y Back-Propagation, y los métodos iterativos: Distorted Born, Contrast Source Inversion Method, etc.<br>
   a. [Método de Born](https://github.com/rirastorza/Intro2MI/blob/main/metodos_deterministicos/metodo_Born.ipynb)<br>
   b. [Método Back-Propagation](https://github.com/rirastorza/Intro2MI/blob/main/metodos_deterministicos/metodo_Backprop.ipynb)<br>

3. Métodos estocásticos

4. Métodos que utilizan Inteligencia Artificial


---

## Recursos
Estas notas recopilan diferentes fuentes mostradas a continuación:

1. Xudong Chen, Computational Methods for Electromagnetic Inverse Scattering.
2. Matteo Pastorino, Microwave Imaging.
