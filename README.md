# brownian-optical-trap-3d
Simulación en GPU de una partícula browniana 3D atrapada en un potencial armónico (trampa óptica) usando ecuaciones de Langevin con PyTorch. Visualización de la trayectoria y difusión térmica.
# Brownian motion in a 3D optical trap – GPU simulation

Simulación de una partícula browniana tridimensional dentro de una trampa óptica (potencial armónico asimétrico), resuelta mediante la **ecuación de Langevin** y acelerada con **PyTorch en GPU**. Se genera una gráfica 3D de la trayectoria completa.

---

## 📘 Física del sistema

La dinámica de una partícula inmersa en un fluido con rozamiento viscoso y sometida a una fuerza externa conservativa y ruido térmico se describe por la ecuación de Langevin sobreamortiguada:

\[
\gamma \frac{d\mathbf{r}}{dt} = -\nabla U(\mathbf{r}) + \sqrt{2\gamma k_B T}\,\boldsymbol{\eta}(t)
\]

donde:
- \(\gamma\) es el coeficiente de fricción (aquí \(\gamma=1\)).
- \(U(\mathbf{r}) = \frac{1}{2}(k_x x^2 + k_y y^2 + k_z z^2)\) es el potencial de la trampa óptica.
- \(k_x, k_y, k_z\) son las rigideces en cada dirección.
- \(D = k_B T / \gamma\) es el coeficiente de difusión térmica (en el código se fija directamente \(D\)).
- \(\boldsymbol{\eta}(t)\) es ruido blanco gaussiano con \(\langle \eta_i(t) \rangle = 0\) y \(\langle \eta_i(t)\eta_j(t') \rangle = \delta_{ij}\delta(t-t')\).

Discretizando con el método de **Euler‑Maruyama**:

\[
\begin{aligned}
x_{t+dt} &= x_t - \frac{k_x}{\gamma} x_t dt + \sqrt{2Ddt}\,\mathcal{N}(0,1) \\
y_{t+dt} &= y_t - \frac{k_y}{\gamma} y_t dt + \sqrt{2Ddt}\,\mathcal{N}(0,1) \\
z_{t+dt} &= z_t - \frac{k_z}{\gamma} z_t dt + \sqrt{2Ddt}\,\mathcal{N}(0,1)
\end{aligned}
\]

La simulación reproduce el movimiento browniano confinado y permite explorar cómo los distintos valores de rigidez y difusión modifican la exploración del espacio 3D.

---

## 🧠 Características del código

- **100 % en GPU** con tensores PyTorch, sin bucles en CPU más allá del paso temporal (el bucle `for` es inevitable pero muy ligero).
- Ruido gaussiano generado directamente en GPU con `torch.randn`.
- Escalado automático de la amplitud de ruido según \( \sqrt{2D dt} \).
- Visualización 3D de la trayectoria con `matplotlib`.
- Compatible con CUDA; la línea `device = "cuda"` selecciona la GPU automáticamente.
- Incluye una corrección para entornos macOS que puedan tener conflictos con la biblioteca MKL de Intel (`KMP_DUPLICATE_LIB_OK`).

---

## 📦 Dependencias

- Python ≥ 3.8
- PyTorch ≥ 1.10 (con soporte CUDA si se desea usar GPU)
- Matplotlib ≥ 3.5
- NumPy (instalado automáticamente con PyTorch)

Instalación rápida con `pip`:

```bash
pip install torch matplotlib
