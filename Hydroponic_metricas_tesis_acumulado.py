import json
import matplotlib.pyplot as plt

# Cargar el archivo JSON
with open("/kaggle/working/Hydroponic_metricas_tesis_acumulado.json", "r") as f:
    datos = json.load(f)

# Extraer historial
historial = datos["historial"]

# Listas para graficar
fechas = [entry["fecha"] for entry in historial]
ph_antes = [entry["datos_completos"]["ph_antes"] for entry in historial]
ph_despues = [entry["datos_completos"]["ph_despues"] for entry in historial]
tds_antes = [entry["datos_completos"]["tds_antes"] for entry in historial]
tds_despues = [entry["datos_completos"]["tds_despues"] for entry in historial]

# Gráfico de pH
plt.figure(figsize=(12, 5))
plt.fill_between(fechas, 5.8, 6.2, color="green", alpha=0.2)
plt.plot(fechas, ph_antes, label="pH antes", marker="o")
plt.plot(fechas, ph_despues, label="pH después", marker="x")
plt.xticks(rotation=45)
plt.xlabel("Fecha")
plt.ylabel("pH")
plt.title("Evolución del pH")
plt.legend()
plt.tight_layout()
plt.show()

# Gráfico de TDS
plt.figure(figsize=(12, 5))
plt.fill_between(fechas, 850, 950, color="green", alpha=0.2)
plt.plot(fechas, tds_antes, label="TDS antes", marker="o")
plt.plot(fechas, tds_despues, label="TDS después", marker="x")
plt.xticks(rotation=45)
plt.xlabel("Fecha")
plt.ylabel("TDS (ppm)")
plt.title("Evolución del TDS")
plt.legend()
plt.tight_layout()
plt.show()
