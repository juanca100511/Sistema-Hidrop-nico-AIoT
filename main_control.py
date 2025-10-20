import time
import os
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_dht
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
from HydroponicEnv import HidroponicoEnv
import datetime
import logging
import subprocess
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import platform
import psutil

# =============================================================
# CONFIGURACIÓN INICIAL
# =============================================================

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/LENOVO/Hydroponic_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Parámetros del sistema
TARGET_PH = 6.0
TARGET_EC = 800.0
PH_LIMITS = (4.5, 8.5)
EC_LIMITS = (300.0, 1800.0)
VMAX_PH = 10.0
VMAX_EC = 20.0
CONTROL_HOUR = 10    # 10:30 AM
CONTROL_MINUTE = 30
SHUTDOWN_HOUR = 11   # 11:00 AM
SHUTDOWN_MINUTE = 0

# =============================================================
# CLASES PARA MONITOREO CIENTÍFICO
# =============================================================

class MetricasTesis:
    """Sistema de recolección de métricas para validación científica"""
    def __init__(self):
        self.inicio_ejecucion = time.time()
        self.datos = {
            'timestamp': [],
            'ph': [],
            'tds': [],
            'temp': [],
            'accion_ph': [],
            'accion_tds': [],
            'ph_error': [],
            'tds_error': [],
            'rendimiento': [],
            'tiempo_ejecucion': [],
            'memoria_usada': [],
            'cpu_uso': [],
            'alertas': []
        }
    
    def registrar(self, ph, tds, temp, accion_ph, accion_tds, ph_error, tds_error):
        """Registra una nueva entrada en las métricas"""
        now = datetime.datetime.now()
        self.datos['timestamp'].append(now.isoformat())
        self.datos['ph'].append(float(ph))
        self.datos['tds'].append(float(tds))
        self.datos['temp'].append(float(temp))
        self.datos['accion_ph'].append(float(accion_ph))
        self.datos['accion_tds'].append(float(accion_tds))
        self.datos['ph_error'].append(float(ph_error))
        self.datos['tds_error'].append(float(tds_error))
        
        # Calcular rendimiento (1 - error normalizado)
        error_total = np.sqrt(ph_error**2 + (tds_error/100)**2)
        error_max = max(0.5, abs(PH_LIMITS[1] - TARGET_PH), abs(EC_LIMITS[1] - TARGET_EC))
        rendimiento = max(0, 1 - (error_total/error_max))
        self.datos['rendimiento'].append(rendimiento)
        
        # Recursos del sistema
        self.datos['tiempo_ejecucion'].append(time.time() - self.inicio_ejecucion)
        self.datos['memoria_usada'].append(psutil.virtual_memory().percent)
        self.datos['cpu_uso'].append(psutil.cpu_percent())
    
    def registrar_alerta(self, tipo, mensaje):
        """Registra una alerta de investigación"""
        self.datos['alertas'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'tipo': tipo,
            'mensaje': mensaje
        })
        logger.warning(f"ALERTA INVESTIGACIÓN ({tipo}): {mensaje}")
    
    def generar_reporte(self):
        """Genera reporte completo en formato JSON (acumulado por días)"""
        archivo = '/home/LENOVO/Hydroponic_metricas_tesis.json'
        datos_actuales = {}
        
        # Leer datos existentes si el archivo ya existe
        if os.path.exists(archivo):
            with open(archivo, 'r') as f:
                try:
                    datos_actuales = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Archivo JSON corrupto. Se creará uno nuevo.")
        
        # Crear estructura de reporte diario
        reporte_diario = {
            'fecha': datetime.datetime.now().strftime("%Y-%m-%d"),
            'metadatos': {
                'sistema': platform.platform(),
                'python_version': platform.python_version(),
                'inicio_ejecucion': datetime.datetime.fromtimestamp(self.inicio_ejecucion).isoformat(),
                'duracion_total': time.time() - self.inicio_ejecucion,
                'configuracion': {
                    'TARGET_PH': TARGET_PH,
                    'TARGET_EC': TARGET_EC,
                    'PH_LIMITS': PH_LIMITS,
                    'EC_LIMITS': EC_LIMITS,
                    'VMAX_PH': VMAX_PH,
                    'VMAX_EC': VMAX_EC
                }
            },
            'estadisticas': {
                'ph_mean': np.mean(self.datos['ph']) if self.datos['ph'] else 0,
                'tds_mean': np.mean(self.datos['tds']) if self.datos['tds'] else 0,
                'rendimiento_promedio': np.mean(self.datos['rendimiento']) if self.datos['rendimiento'] else 0,
                'accion_ph_mean': np.mean(self.datos['accion_ph']) if self.datos['accion_ph'] else 0,
                'accion_tds_mean': np.mean(self.datos['accion_tds']) if self.datos['accion_tds'] else 0,
                'memoria_promedio': np.mean(self.datos['memoria_usada']) if self.datos['memoria_usada'] else 0,
                'cpu_promedio': np.mean(self.datos['cpu_uso']) if self.datos['cpu_uso'] else 0
            },
            'datos_completos': self.datos
        }
        
        # Agregar reporte diario al historial
        if 'historial' not in datos_actuales:
            datos_actuales['historial'] = []
        
        # Verificar si ya existe un reporte para hoy
        hoy = datetime.datetime.now().strftime("%Y-%m-%d")
        historial_actualizado = []
        encontrado = False
        
        for reporte in datos_actuales['historial']:
            if reporte['fecha'] == hoy:
                # Reemplazar el reporte existente de hoy
                historial_actualizado.append(reporte_diario)
                encontrado = True
            else:
                historial_actualizado.append(reporte)
        
        if not encontrado:
            historial_actualizado.append(reporte_diario)
        
        datos_actuales['historial'] = historial_actualizado
        
        # Guardar datos acumulados
        with open(archivo, 'w') as f:
            json.dump(datos_actuales, f, indent=4)
        
        logger.info(f"Reporte diario guardado para {hoy}")
        return reporte_diario

class AnalizadorRendimiento:
    """Herramientas para análisis de rendimiento del sistema"""
    @staticmethod
    def generar_graficos(metricas):
        """Genera gráficos científicos para la tesis"""
        try:
            # Deshabilitar warnings de matplotlib
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            
            plt.figure(figsize=(14, 10))
            
            # Gráfico 1: Evolución de parámetros
            plt.subplot(2, 2, 1)
            plt.plot(metricas.datos['timestamp'], metricas.datos['ph'], 'r-', label='pH')
            plt.plot(metricas.datos['timestamp'], [TARGET_PH]*len(metricas.datos['timestamp']), 'g--', label='Objetivo pH')
            plt.plot(metricas.datos['timestamp'], metricas.datos['tds'], 'b-', label='TDS')
            plt.plot(metricas.datos['timestamp'], [TARGET_EC]*len(metricas.datos['timestamp']), 'c--', label='Objetivo TDS')
            plt.title('Evolución de Parámetros Hidropónicos')
            plt.xlabel('Tiempo')
            plt.ylabel('Valor')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # Gráfico 2: Acciones de control
            plt.subplot(2, 2, 2)
            plt.plot(metricas.datos['timestamp'], metricas.datos['accion_ph'], 'm-', label='Acción pH')
            plt.plot(metricas.datos['timestamp'], metricas.datos['accion_tds'], 'y-', label='Acción TDS')
            plt.title('Acciones de Control')
            plt.xlabel('Tiempo')
            plt.ylabel('Valor de Acción')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # Gráfico 3: Rendimiento y errores
            plt.subplot(2, 2, 3)
            plt.plot(metricas.datos['timestamp'], metricas.datos['rendimiento'], 'g-', label='Rendimiento')
            plt.plot(metricas.datos['timestamp'], metricas.datos['ph_error'], 'r--', label='Error pH')
            plt.plot(metricas.datos['timestamp'], metricas.datos['tds_error'], 'b--', label='Error TDS')
            plt.title('Rendimiento y Errores')
            plt.xlabel('Tiempo')
            plt.ylabel('Valor')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # Gráfico 4: Recursos del sistema
            plt.subplot(2, 2, 4)
            plt.plot(metricas.datos['timestamp'], metricas.datos['cpu_uso'], 'c-', label='Uso CPU (%)')
            plt.plot(metricas.datos['timestamp'], metricas.datos['memoria_usada'], 'm-', label='Uso Memoria (%)')
            plt.title('Consumo de Recursos')
            plt.xlabel('Tiempo')
            plt.ylabel('Porcentaje')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Guardar gráfico en archivo (con fecha)
            fecha = datetime.datetime.now().strftime("%Y-%m-%d")
            plt.savefig(f'/home/LENOVO/Hydroponic_graficos_rendimiento_{fecha}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Gráficos de rendimiento generados correctamente")
            return True
        except Exception as e:
            logger.error(f"Error generando gráficos: {str(e)}")
            return False

    @staticmethod
    def exportar_csv(metricas):
        """Exporta datos a CSV para análisis estadístico (acumulando días)"""
        archivo = '/home/LENOVO/Hydroponic_datos_tesis.csv'
        
        # Crear copia de los datos excluyendo 'alertas'
        datos_para_csv = {k: v for k, v in metricas.datos.items() if k != 'alertas'}
        df_nuevo = pd.DataFrame(datos_para_csv)
        
        # Agregar columna de fecha
        fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d")
        df_nuevo['fecha'] = fecha_actual
        
        try:
            # Leer datos existentes si el archivo ya existe
            if os.path.exists(archivo):
                df_existente = pd.read_csv(archivo)
                
                # Filtrar datos de hoy si existen
                hoy = datetime.datetime.now().strftime("%Y-%m-%d")
                # Conservar solo los registros que no son de hoy
                df_existente = df_existente[df_existente['fecha'] != hoy]
                
                # Combinar con los nuevos datos
                df_completo = pd.concat([df_existente, df_nuevo], ignore_index=True)
            else:
                df_completo = df_nuevo
            
            # Guardar datos acumulados
            df_completo.to_csv(archivo, index=False)
            logger.info("Datos exportados a CSV correctamente (acumulados)")
            return True
        except Exception as e:
            logger.error(f"Error exportando a CSV: {str(e)}")
            return False

class ValidadorCientifico:
    """Validaciones científicas para integridad de datos"""
    @staticmethod
    def validar_sensores():
        """Valida consistencia de lecturas de sensores"""
        lecturas = []
        logger.info("Iniciando validación científica de sensores...")
        
        for i in range(5):  # Tomar 5 muestras
            try:
                ph, tds, temp = leer_sensores()
                if None in (ph, tds, temp):
                    logger.error(f"Error en lectura {i+1}/5: valores nulos")
                    return False
                
                lecturas.append({
                    'ph': ph,
                    'tds': tds,
                    'temp': temp,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                time.sleep(1)
            except Exception as e:
                logger.error(f"Excepción en validación de sensores: {str(e)}")
                return False
        
        # Calcular estadísticas
        ph_values = [r['ph'] for r in lecturas]
        tds_values = [r['tds'] for r in lecturas]
        
        media_ph = np.mean(ph_values)
        std_ph = np.std(ph_values)
        media_tds = np.mean(tds_values)
        std_tds = np.std(tds_values)
        
        # Umbrales de validación científica
        valido = True
        if std_ph > 0.1:
            logger.error(f"Alta desviación estándar en pH: {std_ph:.4f}")
            valido = False
            
        if std_tds > 10:
            logger.error(f"Alta desviación estándar en TDS: {std_tds:.2f}")
            valido = False
            
        if abs(media_ph - TARGET_PH) > 2.0:
            logger.error(f"Media de pH fuera de rango esperado: {media_ph:.2f}")
            valido = False
            
        if abs(media_tds - TARGET_EC) > 300:
            logger.error(f"Media de TDS fuera de rango esperado: {media_tds:.2f}")
            valido = False
        
        if valido:
            logger.info("Validación científica de sensores exitosa")
        else:
            logger.error("Validación científica de sensores fallida")
            
        return valido

    @staticmethod
    def detectar_anomalias(ph, tds, metricas):
        """Detecta anomalias científicamente significativas"""
        try:
            anomalias = []
            
            # Detección básica de rango físico
            if ph < PH_LIMITS[0] or ph > PH_LIMITS[1]:
                anomalias.append(f"pH fuera de rango físico: {ph:.2f}")
                
            if tds < EC_LIMITS[0] or tds > EC_LIMITS[1]:
                anomalias.append(f"TDS fuera de rango físico: {tds:.0f}")
            
            # Verificar datos históricos
            if metricas and 'ph' in metricas.datos and len(metricas.datos['ph']) > 1:
                ultimo_ph = metricas.datos['ph'][-1]
                cambio_ph = abs(ph - ultimo_ph)
                if cambio_ph > 0.5:
                    anomalias.append(f"Cambio abrupto en pH: Δ={cambio_ph:.2f}")
            
            if metricas and 'tds' in metricas.datos and len(metricas.datos['tds']) > 1:
                ultimo_tds = metricas.datos['tds'][-1]
                cambio_tds = abs(tds - ultimo_tds)
                if cambio_tds > 100:
                    anomalias.append(f"Cambio abrupto en TDS: Δ={cambio_tds:.0f}")
            
            return anomalias
        except Exception as e:
            logger.error(f"Error en detección de anomalias: {str(e)}")
            return []

# =============================================================
# CONFIGURACIÓN DE HARDWARE
# =============================================================
# Configuración de GPIO
GPIO.setmode(GPIO.BCM)
BOMBA_ACIDO = 16
BOMBA_BASE = 17
BOMBA_NUTRIENTES = 19
BOMBA_AGUA = 20
GPIO.setup(BOMBA_ACIDO, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(BOMBA_BASE, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(BOMBA_NUTRIENTES, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(BOMBA_AGUA, GPIO.OUT, initial=GPIO.HIGH)
# Inicializar sensores
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads_tds = ADS.ADS1015(i2c, address=0x49)
    ads_ph = ADS.ADS1015(i2c, address=0x48)
    sensor_tds = AnalogIn(ads_tds, ADS.P0)
    sensor_ph = AnalogIn(ads_ph, ADS.P0)
    dht_device = adafruit_dht.DHT11(board.D4)
    lcd = CharLCD(i2c_expander="PCF8574",address=0x27, port=1, cols=16, rows=2)
    logger.info("Sensores y LCD inicializados")
except Exception as e:
    logger.error(f"Error inicializando hardware: {e}")
    sys.exit(1)
# =============================================================
# CONFIGURACIÓN DE SOFTWARE
# =============================================================
# Cargar modelo y normalizador
try:
    logger.info("Cargando modelo y normalizador...")
    dummy_env = DummyVecEnv([lambda: HidroponicoEnv()])
    model = SAC.load("/home/LENOVO/sac_hidroponico_model.zip")
    normalizer = VecNormalize.load("/home/LENOVO/hidroponico_vecnormalize.pkl", dummy_env)
    normalizer.training = False
    logger.info("Modelo y normalizador cargados correctamente")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")
    sys.exit(1)

# =============================================================
# FUNCIONES PRINCIPALES DEL SISTEMA
# =============================================================

# Funciones de calibración
def voltage_to_ph(v):
    return max(0, min(14, -1.58 * v + 8.91 - 0.5))

def voltage_to_tds(v):
    if v < 0.5:
        return max(0, 500 * v - 50)
    elif v < 1.0:
        return 600 * v - 70
    elif v < 2.2:
        return 500 * v + 50
    else:
        return 900 * v - 800

def leer_sensores():
    """Lee los sensores con manejo de errores robusto"""
    try:
        raw_ph = sensor_ph.voltage
        raw_tds = sensor_tds.voltage
        temperatura = dht_device.temperature or 25.0
        
        ph = voltage_to_ph(raw_ph)
        tds = voltage_to_tds(raw_tds)
        
        # Aplicar límites físicos
        ph = np.clip(ph, *PH_LIMITS)
        tds = np.clip(tds, *EC_LIMITS)
        
        logger.info(f"Lectura sensores: pH={ph:.2f}, TDS={tds:.0f}, Temp={temperatura:.1f}°C")
        return ph, tds, temperatura
    except Exception as e:
        logger.error(f"Error leyendo sensores: {e}")
        return None, None, None

def activar_bomba(pin, nombre, accion, max_ml, ml_por_segundo=1.0):
    """
    Activa una bomba en base a la acción predicha y los parámetros calibrados.

    Parámetros:
    - pin: GPIO asignado a la bomba.
    - nombre: string descriptivo de la bomba.
    - accion: valor continuo [-1,1] que viene del modelo (dirección/intensidad).
    - eff_param: eficiencia del aditivo (ΔpH o ΔEC por mL).
    - f: caudal real de la bomba (mL/s), obtenido experimentalmente.
    - t_delay: tiempo muerto de la bomba (s), obtenido experimentalmente.
    - delta_objetivo: cambio requerido en la variable (ΔpH o ΔEC).

    Procedimiento:
    1. Calcular volumen requerido (mL).
    2. Convertir volumen a tiempo bruto de activación.
    3. Ajustar con tiempo muerto.
    4. Activar y desactivar la bomba.
    """

    try:
        # 1. Volumen requerido según eficiencia
        V_req = abs(delta_objetivo) / eff_param   # mL

        # 2. Tiempo bruto de operación
        t_on_raw = V_req / f  # segundos

        # 3. Ajuste por tiempo muerto
        t_on = max(0, t_on_raw + t_delay)

        # 4. Activar bomba si el modelo lo indica (accion > 0)
        if accion > 0 and t_on > 0:
            print(f"[INFO] Activando {nombre} durante {t_on:.2f} s (Δ objetivo = {delta_objetivo})")
            GPIO.output(pin, GPIO.LOW)   # encender relé
            time.sleep(t_on)
            GPIO.output(pin, GPIO.HIGH)  # apagar relé
            print(f"[INFO] {nombre} desactivada.")
        else:
            print(f"[INFO] {nombre}: No se requiere activación. (Δ objetivo = {delta_objetivo})")

    except Exception as e:
        print(f"[ERROR] en {nombre}: {e}")
        GPIO.output(pin, GPIO.HIGH)  # seguridad: apagar bomba


def politica_seguridad(ph, tds):
    """Política de seguridad para estados nunca vistos"""
    logger.warning("Activando política de seguridad")
    
    # Calcular desviaciones
    desv_ph = ph - TARGET_PH
    desv_tds = tds - TARGET_EC
    
    # Determinar acciones conservadoras
    accion_pH = 0.0
    accion_TDS = 0.0
    
    # Corrección de pH
    if desv_ph > 0.5:  # pH demasiado alto
        accion_pH = -0.3  # Agregar ácido
    elif desv_ph < -0.5:  # pH demasiado bajo
        accion_pH = 0.3  # Agregar base
    
    # Corrección de TDS
    if desv_tds > 200:  # TDS demasiado alto
        accion_TDS = -0.3  # Agregar agua
    elif desv_tds < -200:  # TDS demasiado bajo
        accion_TDS = 0.3  # Agregar nutrientes
    
    logger.info(f"Acción seguridad: pH={accion_pH:.2f}, TDS={accion_TDS:.2f}")
    return np.array([accion_pH, accion_TDS], dtype=np.float32)

def apagar_sistema(metricas):
    """Apaga el sistema de manera segura con generación de reportes"""
    logger.info("Iniciando secuencia de apagado científico")
    
    try:
        # Generar reportes finales
        logger.info("Generando reportes científicos finales...")
        reporte = metricas.generar_reporte()
        AnalizadorRendimiento.exportar_csv(metricas)
        AnalizadorRendimiento.generar_graficos(metricas)
        
        # Resumen ejecutivo
        logger.info("="*60)
        logger.info("RESUMEN EJECUTIVO PARA TESIS")
        logger.info(f"Duración total: {reporte['metadatos']['duracion_total']:.1f} segundos")
        logger.info(f"Rendimiento promedio: {reporte['estadisticas']['rendimiento_promedio']*100:.1f}%")
        logger.info(f"pH promedio: {reporte['estadisticas']['ph_mean']:.2f} (Objetivo: {TARGET_PH})")
        logger.info(f"TDS promedio: {reporte['estadisticas']['tds_mean']:.0f} (Objetivo: {TARGET_EC})")
        logger.info(f"Alertas registradas: {len(metricas.datos['alertas'])}")
        logger.info("="*60)
        
        # Limpiar recursos
        GPIO.cleanup()
        lcd.clear()
        lcd.close()
        logger.info("Recursos liberados")
        
        # Apagar Raspberry Pi
        subprocess.run(["sudo", "shutdown", "-h", "now"])
    except Exception as e:
        logger.error(f"Error en secuencia de apagado: {e}")
    sys.exit(0)

def ejecutar_ciclo_control(metricas):
    """Ejecuta un ciclo completo de control con instrumentación científica"""
    # Variables para seguimiento de errores
    ema_ph_error = 0.0
    ema_ec_error = 0.0
    
    # Validación científica inicial
    if not ValidadorCientifico.validar_sensores():
        metricas.registrar_alerta("SENSORES", "Validación científica fallida")
    
    # Leer sensores
    ph, tds, temp = leer_sensores()
    if ph is None or tds is None:
        logger.error("Error crítico en sensores. Apagando sistema.")
        metricas.registrar_alerta("CRITICO", "Fallo en lectura de sensores")
        apagar_sistema(metricas)
    
    # Detección de anomalías
    anomalias = ValidadorCientifico.detectar_anomalias(ph, tds, metricas)
    for anomalia in anomalias:
        metricas.registrar_alerta("ANOMALIA", anomalia)
    
    # Calcular errores actuales
    ph_error = abs(ph - TARGET_PH)
    ec_error = abs(tds - TARGET_EC)
    
    # Actualizar errores acumulados (EMA)
    ema_ph_error = 0.85 * ema_ph_error + 0.15 * ph_error
    ema_ec_error = 0.85 * ema_ec_error + 0.15 * ec_error
    
    # Crear estado
    estado = np.array([ph, tds, ema_ph_error, ema_ec_error], dtype=np.float32)
    logger.info(f"Estado: pH={estado[0]:.2f}, TDS={estado[1]:.0f}, "
                f"ePH_acum={estado[2]:.2f}, eEC_acum={estado[3]:.2f}")
    
    # Verificar si el estado es extremo
    estado_extremo = (
        ph < PH_LIMITS[0] + 0.5 or 
        ph > PH_LIMITS[1] - 0.5 or 
        tds < EC_LIMITS[0] + 100 or 
        tds > EC_LIMITS[1] - 100
    )
    
    # Predecir acción
    if estado_extremo:
        accion = politica_seguridad(ph, tds)
        metricas.registrar_alerta("ESTADO_EXTREMO", "Activada política de seguridad")
    else:
        try:
            # Normalizar estado
            estado_normalizado = normalizer.normalize_obs(estado.reshape(1, -1))[0]
            
            # Predecir acción
            accion, _ = model.predict(estado_normalizado, deterministic=True)
            
            # Verificar si la acción es extrema (posible estado no visto)
            if np.any(np.abs(accion) > 1.5):
                logger.warning("Acción extrema detectada. Usando política de seguridad")
                metricas.registrar_alerta("ACCION_EXTREMA", f"Valores: {accion}")
                accion = politica_seguridad(ph, tds)
        except Exception as e:
            logger.error(f"Error en predicción: {e}. Usando política de seguridad")
            metricas.registrar_alerta("MODELO", f"Error en predicción: {str(e)}")
            accion = politica_seguridad(ph, tds)
    
    accion_pH, accion_TDS = accion
    logger.info(f"Acción final: pH={accion_pH:.2f}, TDS={accion_TDS:.2f}")
    
    # Registrar en métricas científicas
    metricas.registrar(ph, tds, temp, accion_pH, accion_TDS, ph_error, ec_error)
    
    # Mostrar en LCD
    mostrar_en_lcd(ph, tds, accion_pH, accion_TDS)
    
    # Activar bombas de forma secuencial con tiempo de mezcla
    if accion_pH < 0:
        activar_bomba(BOMBA_ACIDO, "ÁCIDO", accion_pH, max_ml=VMAX_PH)
    elif accion_pH > 0:
        activar_bomba(BOMBA_BASE, "BASE", accion_pH, max_ml=VMAX_PH)
    
    if accion_TDS < 0:
        activar_bomba(BOMBA_AGUA, "AGUA", accion_TDS, max_ml=VMAX_EC)
    elif accion_TDS > 0:
        activar_bomba(BOMBA_NUTRIENTES, "NUTRIENTES", accion_TDS, max_ml=VMAX_EC)

# =============================================================
# FUNCIÓN PRINCIPAL
# =============================================================

def main():
    # Inicializar sistema de metricas para tesis
    metricas = MetricasTesis()
    
    try:
        logger.info("Sistema de Control Hidroponico Cientifico Iniciado")
        logger.info("="*60)
        logger.info("PARAMETROS DE INVESTIGACION")
        logger.info(f"Objetivo pH: {TARGET_PH} | Objetivo TDS: {TARGET_EC}")
        logger.info(f"Limites pH: {PH_LIMITS} | Limites TDS: {EC_LIMITS}")
        logger.info("="*60)
        
        # Calcular tiempo hasta las 10:30 AM
        now = datetime.datetime.now()
        target_time = now.replace(hour=CONTROL_HOUR, minute=CONTROL_MINUTE, second=0, microsecond=0)
        
        # Si ya es despues de 10:30, ejecutar inmediatamente
        if now >= target_time:
            logger.warning("Hora de control ya paso! Ejecutando inmediatamente")
            ejecutar_ciclo_control(metricas)
        else:
            # Esperar hasta las 10:30
            wait_seconds = (target_time - now).total_seconds()
            logger.info(f"Esperando {wait_seconds/60:.1f} minutos hasta las 10:30 AM...")
            time.sleep(wait_seconds)
            
            # Ejecutar control cientifico
            logger.info("Iniciando ciclo de control cientifico")
            ejecutar_ciclo_control(metricas)
        
        # Calcular tiempo restante hasta las 11:00
        now = datetime.datetime.now()
        shutdown_time = now.replace(hour=SHUTDOWN_HOUR, minute=SHUTDOWN_MINUTE, second=0, microsecond=0)
        wait_seconds = (shutdown_time - now).total_seconds()
        
        # Guardar reportes finales ANTES del apagado
        logger.info("Guardando reportes cientificos finales...")
        metricas.generar_reporte()
        AnalizadorRendimiento.exportar_csv(metricas)
        AnalizadorRendimiento.generar_graficos(metricas)
        
        if wait_seconds > 0:
            logger.info(f"Esperando {wait_seconds} segundos hasta apagado externo...")
            time.sleep(wait_seconds)
        
        logger.info("Operacion completada. Esperando apagado por temporizador externo.")
        
        # Limpiar recursos sin apagar (el temporizador cortara la energia)
        GPIO.cleanup()
        lcd.clear()
        lcd.close()
        
        # Esperar indefinidamente (el corte de energia vendra pronto)
        while True:
            time.sleep(10)
            
    except Exception as e:
        logger.critical(f"Error critico no manejado: {e}", exc_info=True)
        # Intentar guardar metricas antes del apagado
        try:
            metricas.generar_reporte()
        except:
            pass
        # Limpiar recursos
        GPIO.cleanup()
        lcd.clear()
        lcd.close()
        
if __name__ == "__main__":
    main()