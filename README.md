# 🌿 Sistema Hidropónico AIoT Controlado por Inteligencia Artificial (SAC)

Este proyecto implementa un **sistema hidropónico autónomo** basado en **Inteligencia Artificial (SAC – Soft Actor-Critic)** e **Internet de las Cosas (AIoT)**, desarrollado en **Raspberry Pi** con sensores de pH, TDS y temperatura.  
El sistema permite el **control inteligente y adaptativo de la solución nutritiva**, optimizando el uso del agua y la energía en entornos rurales con limitaciones de conectividad.

---

## 🧩 Descripción general

El sistema combina hardware, software y aprendizaje por refuerzo para crear un entorno hidropónico autónomo, capaz de mantener parámetros ideales de pH y conductividad eléctrica (TDS) sin supervisión constante.  
El control se ejecuta localmente en la Raspberry Pi mediante un modelo SAC previamente entrenado en un entorno de simulación.

**Componentes principales:**
- **Raspberry Pi 4B** – Nodo de control y procesamiento.
- **Sensores:** pH, TDS, temperatura (DHT11).
- **Actuadores:** Bombas peristálticas (ácido, base, nutrientes, agua).
- **Módulos:** ADS1015 (lectura analógica), relé 4 canales, LCD I2C (PCF8574).
- **Energía:** Alimentación solar con batería de respaldo de 12V.
- **Software:** Python 3.10, Stable-Baselines3 (SAC), Matplotlib, Pandas, RPi.GPIO.

---

## 🧱 Arquitectura general del sistema

El proyecto se organiza en tres capas principales:

1. **Capa de hardware:** Sensores, actuadores y sistema de energía solar.  
2. **Capa de control local (Edge AI):** Raspberry Pi con el modelo SAC entrenado.  
3. **Capa de análisis y registro:** Recolección de métricas, reportes JSON/CSV y visualización científica.

```mermaid
flowchart TB
    subgraph Energía Solar
        A[☀️ Panel Solar] --> B[🔋 Controlador de Carga]
        B --> C[🔋 Batería 12V]
        C --> D[⚙️ Regulador DC-DC 5V]
    end

    subgraph Electrónica
        D --> E[🖥️ Raspberry Pi (SAC)]
        E --> F[📈 ADS1015]
        F --> G[🌿 Sensores pH/TDS]
        E --> H[📟 LCD I2C]
        E --> I[🔌 Relé 4 Canales]
        I --> J[💧 Bombas Peristálticas]
    end

    E --> K[📊 Registro y Métricas JSON/CSV]
