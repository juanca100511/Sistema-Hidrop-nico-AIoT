<h1 align="center">🌿 Sistema Hidropónico AIoT con Inteligencia Artificial (SAC)</h1>

<p align="center">
  <b>Gestión Autónoma de Viveros Hidropónicos mediante Inteligencia Artificial de las Cosas</b><br>
  Optimización del uso del agua en zonas rurales mediante control inteligente y energía solar.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
  <img src="https://img.shields.io/badge/Raspberry%20Pi-4B-red?logo=raspberrypi">
  <img src="https://img.shields.io/badge/AI-SAC%20Algorithm-green?logo=tensorflow">
  <img src="https://img.shields.io/badge/License-MIT-yellow?logo=open-source-initiative">
</p>

---

## 🧩 Descripción general

Este proyecto implementa un **sistema hidropónico autónomo AIoT** basado en **Inteligencia Artificial (SAC – Soft Actor-Critic)** e **Internet de las Cosas**, desarrollado en **Raspberry Pi**.  
Permite **monitorear y controlar parámetros críticos (pH, TDS, temperatura)**, utilizando energía solar y actuadores peristálticos para mantener condiciones óptimas de cultivo.

💧 **Objetivo:** optimizar el uso del recurso hídrico y energético en entornos rurales con limitaciones de conectividad.  
📍 **Validado en:** Comunidad de Mantoclla, distrito de Anta, Cusco – Perú.

---

## ⚙️ Arquitectura del sistema


    subgraph Alimentación Solar
        A[☀️ Panel Solar] --> B[🔋 Controlador de carga]
        B --> C[🔋 Batería 12V]
        C --> D[⚙️ Regulador DC-DC 5V]
    end

    subgraph Electrónica AIoT
        D --> E[🖥️ Raspberry Pi 4B\n(SAC AI Controller)]
        E --> F[📈 Módulo ADS1015\nLectura analógica I2C]
        F --> G[🌿 Sensores pH / TDS / DHT11]
        E --> H[📟 LCD I2C (PCF8574)]
        E --> I[🔌 Módulo Relé 4 canales]
        I --> J[💧 Bombas peristálticas + Bomba recirculación]
    end

