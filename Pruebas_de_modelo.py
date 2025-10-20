from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import gym

# Crear entorno Gymnasium mínimo
class HidroponicoEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def reset(self):
        return np.zeros(4, dtype=np.float32)
    
    def step(self, action):
        return self.reset(), 0, False, {}

# Clase de control mejorada con manejo de valores extremos
class HidroponicoController:
    def __init__(self, model_path, normalizer_path):
        # Crear entorno válido
        self.env = DummyVecEnv([lambda: HidroponicoEnv()])
        
        # Cargar modelo y normalizador
        self.model = SAC.load(model_path)
        self.normalizer = VecNormalize.load(normalizer_path, self.env)
        self.normalizer.training = False
        
        # Parámetros del sistema (coinciden con el entrenamiento)
        self.ph_obj = 6.0
        self.ec_obj = 800.0
        self.Vmax_ph = 10.0
        self.Vmax_ec = 20.0
        
        # Rangos de operación segura
        self.PH_MIN_SAFE = 4.0
        self.PH_MAX_SAFE = 8.0
        self.EC_MIN_SAFE = 200.0
        self.EC_MAX_SAFE = 2500.0
        
        # Rangos objetivo
        self.PH_MIN_TARGET = 5.5
        self.PH_MAX_TARGET = 6.5
        self.EC_MIN_TARGET = 600.0
        self.EC_MAX_TARGET = 1200.0

    def predict_action(self, ph, ec):
        """Predice acción con manejo de valores extremos"""
        # Verificar y manejar valores fuera de rango seguro
        if ph < self.PH_MIN_SAFE or ph > self.PH_MAX_SAFE:
            print(f"⚠️ ALERTA: pH={ph} está fuera del rango seguro [{self.PH_MIN_SAFE}-{self.PH_MAX_SAFE}]")
            # Acciones conservadoras para valores extremos
            dV_ph = 0.0
            if ph < self.PH_MIN_SAFE:
                dV_ph = self.Vmax_ph  # Agregar base para aumentar pH
            else:
                dV_ph = -self.Vmax_ph  # Agregar ácido para disminuir pH
                
            return [0, 0], dV_ph, 0.0
        
        if ec < self.EC_MIN_SAFE or ec > self.EC_MAX_SAFE:
            print(f"⚠️ ALERTA: EC={ec} está fuera del rango seguro [{self.EC_MIN_SAFE}-{self.EC_MAX_SAFE}]")
            # Acciones conservadoras para valores extremos
            dV_ec = 0.0
            if ec < self.EC_MIN_SAFE:
                dV_ec = self.Vmax_ec  # Agregar nutrientes
            else:
                dV_ec = -self.Vmax_ec  # Agregar agua destilada
                
            return [0, 0], 0.0, dV_ec
        
        # Calcular errores si está en rango seguro
        error_ph = abs(ph - self.ph_obj)
        error_ec = abs(ec - self.ec_obj)
        
        # Construir estado
        state = np.array([ph, ec, error_ph, error_ec], dtype=np.float32)
        
        # Normalizar y predecir
        norm_state = self.normalizer.normalize_obs(state.reshape(1, -1))
        action, _ = self.model.predict(norm_state, deterministic=True)
        
        # Convertir a volúmenes
        dV_ph = action[0][0] * self.Vmax_ph
        dV_ec = action[0][1] * self.Vmax_ec
        
        return action[0], dV_ph, dV_ec

# Función para interpretar acciones
def interpretar_accion(dV_ph, dV_ec):
    """Convierte volúmenes en instrucciones prácticas"""
    if dV_ph > 0:
        ph_txt = f"Agregar {abs(dV_ph):.1f}ml de BASE"
    elif dV_ph < 0:
        ph_txt = f"Agregar {abs(dV_ph):.1f}ml de ÁCIDO"
    else:
        ph_txt = "No ajustar pH"
    
    if dV_ec > 0:
        ec_txt = f"Agregar {abs(dV_ec):.1f}ml de NUTRIENTES"
    elif dV_ec < 0:
        ec_txt = f"Agregar {abs(dV_ec):.1f}ml de AGUA DESTILADA"
    else:
        ec_txt = "No ajustar EC"
    
    return ph_txt, ec_txt

# Información sobre rangos aceptables
def mostrar_rangos():
    print("\n💡 INFORMACIÓN SOBRE RANGOS ACEPTABLES:")
    print(f"pH Óptimo: {5.5}-{6.5} (Objetivo: 6.0)")
    print(f"EC/TDS Óptimo: {600}-{1200} µS/cm (Objetivo: 800 µS/cm)")
    print(f"Rango Seguro pH: {4.0}-{8.0}")
    print(f"Rango Seguro EC: {200}-{2500} µS/cm")
    print("Valores fuera de estos rangos requieren atención especial\n")

# Ejecución principal
if __name__ == "__main__":
    # Configuración (ajustar rutas según sea necesario)
    MODEL_PATH = "sac_hidroponico_model.zip"
    NORMALIZER_PATH = "hidroponico_vecnormalize.pkl"
    
    # Inicializar controlador
    controller = HidroponicoController(MODEL_PATH, NORMALIZER_PATH)
    
    # Mostrar información de rangos
    mostrar_rangos()
    
    # Entrada de usuario con valores por defecto variables
    try:
        ph_input = input("pH actual (ENTER para ejemplos): ")
        ec_input = input("EC actual (ENTER para ejemplos): ")
        
        if not ph_input or not ec_input:
            # Ejemplos que cubren diferentes escenarios
            ejemplos = [
                {"ph": 5.85, "ec": 780.0, "desc": "Valor normal"},
                {"ph": 3.5, "ec": 100.0, "desc": "pH extremadamente bajo, EC muy bajo"},
                {"ph": 8.5, "ec": 4000.0, "desc": "pH extremadamente alto, EC muy alto"},
                {"ph": 5.2, "ec": 1500.0, "desc": "EC alto"},
                {"ph": 7.0, "ec": 500.0, "desc": "pH alto, EC bajo"}
            ]
            print("\n💎 Ejecutando casos de prueba:")
            for i, ejemplo in enumerate(ejemplos):
                ph_actual = ejemplo["ph"]
                ec_actual = ejemplo["ec"]
                print(f"\n🔍 CASO {i+1}: {ejemplo['desc']}")
                print(f"   pH={ph_actual}, EC={ec_actual}")
                
                # Predecir acción
                raw_action, dV_ph, dV_ec = controller.predict_action(ph_actual, ec_actual)
                ph_txt, ec_txt = interpretar_accion(dV_ph, dV_ec)
                
                # Mostrar resultados
                print("="*50)
                if abs(raw_action[0]) > 0.001 or abs(raw_action[1]) > 0.001:
                    print(f"⚡ ACCIÓN DEL MODELO: [{raw_action[0]:.4f}, {raw_action[1]:.4f}]")
                print(f"🧪 AJUSTE pH: {dV_ph:+.2f}ml -> {ph_txt}")
                print(f"🧫 AJUSTE EC: {dV_ec:+.2f}ml -> {ec_txt}")
                print("="*50)
        else:
            ph_actual = float(ph_input)
            ec_actual = float(ec_input)
            
            # Predecir acción
            raw_action, dV_ph, dV_ec = controller.predict_action(ph_actual, ec_actual)
            ph_txt, ec_txt = interpretar_accion(dV_ph, dV_ec)
            
            # Mostrar resultados
            print("\n" + "="*50)
            print(f"⚡ ACCIÓN DEL MODELO: [{raw_action[0]:.4f}, {raw_action[1]:.4f}]")
            print("="*50)
            print(f"🧪 AJUSTE pH: {dV_ph:+.2f}ml -> {ph_txt}")
            print(f"🧫 AJUSTE EC: {dV_ec:+.2f}ml -> {ec_txt}")
            print("="*50)
            
    except Exception as e:
        print(f"Error: {str(e)}")