# env.py
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class HidroponicoEnv(gym.Env):
    """
    Entorno Hidropónico para 40 litros con efectos diferenciados
    - Ácido (pH-), Base (pH+) y Agua Destilada (EC-)
    Estado: [pH, EC, error_pH_acumulado, error_EC_acumulado]
    Acción: [a_pH, a_EC] en [-1,1]^2
    Compatible con Gym v0.21+
    """

    def __init__(
        self,
        ph_obj=6.10,
        ec_obj=800.0,
        Vmax_ph=2.0,
        Vmax_ec=2.0,
        ph_limits=(4.5, 8.5),
        ec_limits=(300.0, 1800.0),
        max_steps=200
    ):
        super().__init__()
        # Parámetros fijos para 40L
        self.VOLUMEN = 40.0  # Litros
        
        # Parámetros de control
        self.ph_obj = ph_obj
        self.ec_obj = ec_obj
        self.Vmax_ph = Vmax_ph
        self.Vmax_ec = Vmax_ec
        
        # Factores de conversión ajustados para 40L
        self.k_ph_acid = 0.0195    # pH por ml de ácido (disminuye pH)
        self.k_ph_base = 0.00194   # pH por ml de base (aumenta pH)
        self.k_ec_nutrients = 8.125  # ppm por ml de nutrientes (aumenta EC)
        self.k_agua_destilada = 2.120  # ppm reducidas por ml de agua destilada (disminuye EC)
        
        # Rangos físicos
        self.ph_limits = ph_limits
        self.ec_limits = ec_limits
        self.max_steps = max_steps
        self.current_step = 0

        # Generador de números aleatorios
        self.np_random = None
        
        # Espacios de observación
        self.observation_space = spaces.Box(
            low=np.array([ph_limits[0], ec_limits[0], 0, 0], dtype=np.float32),
            high=np.array([ph_limits[1], ec_limits[1], 5.0, 500.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Inicializar el generador de números aleatorios
        self.seed()

    def seed(self, seed=None):
        # Implementación del método seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        # Inicializar generador si se proporciona semilla
        if seed is not None:
            self.seed(seed)
            
        # Estado inicial alrededor del objetivo
        self.ph = np.clip(self.np_random.normal(self.ph_obj, 0.3), *self.ph_limits)
        self.ec = np.clip(self.np_random.normal(self.ec_obj, 75), *self.ec_limits)
        self.error_ph_acum = 0.0
        self.error_ec_acum = 0.0
        self.current_step = 0
        
        # Devolver estado e información vacía
        return np.array([
            self.ph, 
            self.ec,
            self.error_ph_acum,
            self.error_ec_acum
        ], dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        a_ph, a_ec = action
        
        # ===== DINÁMICA PARA 40L CON EFECTOS DIFERENCIADOS =====
        # Conversión acción a volumen (con ruido de implementación)
        dV_ph = a_ph * self.Vmax_ph * self.np_random.uniform(0.95, 1.05)
        dV_ec = a_ec * self.Vmax_ec * self.np_random.uniform(0.95, 1.05)
        
        # 1. Ajuste de pH (con efectos diferentes para ácido/base)
        if dV_ph > 0:
            # Agregar BASE (aumenta pH)
            self.ph += self.k_ph_base * dV_ph
        else:
            # Agregar ÁCIDO (disminuye pH)
            self.ph += self.k_ph_acid * dV_ph  # dV_ph negativo = disminución
        
        # 2. Ajuste de EC (con efectos diferentes para nutrientes/agua destilada)
        if dV_ec > 0:
            # Agregar NUTRIENTES (aumenta EC)
            self.ec += self.k_ec_nutrients * dV_ec
        else:
            # Agregar AGUA DESTILADA (disminuye EC)
            self.ec += self.k_agua_destilada * dV_ec  # dV_ec negativo = disminución
        
        # Perturbaciones naturales
        self.ph += self.np_random.normal(0, 0.008)  # Deriva natural
        self.ec -= self.np_random.exponential(0.8)  # Consumo por plantas
        self.ec += self.np_random.normal(0, 3.0)    # Otras variaciones
        
        # Aplicar límites físicos
        self.ph = np.clip(self.ph, *self.ph_limits)
        self.ec = np.clip(self.ec, *self.ec_limits)
        
        # ===== CÁLCULO DE RECOMPENSA =====
        # Errores actuales
        ph_error = abs(self.ph - self.ph_obj)
        ec_error = abs(self.ec - self.ec_obj)
        
        # Actualizar errores acumulados (EMA)
        self.error_ph_acum = 0.85 * self.error_ph_acum + 0.15 * ph_error
        self.error_ec_acum = 0.85 * self.error_ec_acum + 0.15 * ec_error
        
        # Componentes de recompensa
        R_error = - (ph_error**2 + (ec_error/100)**2) * 2
        R_stability = - (self.error_ph_acum + self.error_ec_acum/50)
        R_action = -0.05 * (a_ph**2 + a_ec**2)
        
        # Bonus por control preciso
        bonus = 0.0
        if ph_error < 0.05 and ec_error < 10:
            bonus = 2.0
        
        # Penalización por acciones extremas
        penalty = 0.0
        if abs(a_ph) > 0.8 or abs(a_ec) > 0.8:
            penalty = -0.5
        
        reward = R_error + R_stability + R_action + bonus + penalty
        
        # ===== TERMINACIÓN =====
        terminated = self.current_step >= self.max_steps
        truncated = False  # No hay truncamiento en este entorno
        
        obs = np.array([
            self.ph,
            self.ec,
            self.error_ph_acum,
            self.error_ec_acum
        ], dtype=np.float32)
        
        info = {
            'ph': self.ph,
            'ec': self.ec,
            'action_ph': dV_ph,
            'action_ec': dV_ec,
            'ph_error': ph_error,
            'ec_error': ec_error
        }
        
        # Devolver 5 valores para compatibilidad con Gym v0.26+
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        # Implementación básica de visualización
        if mode == 'human':
            print(f"Step: {self.current_step:3d} | pH: {self.ph:.3f} (obj: {self.ph_obj}) | EC: {self.ec:.1f} (obj: {self.ec_obj})")