from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Crear entorno vectorizado y normalizado
env = make_vec_env(lambda: HidroponicoEnv(), n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Configurar modelo SAC
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=2.5e-4,
    buffer_size=200000,
    batch_size=256,
    ent_coef='auto',
    verbose=1,
    tensorboard_log="./hidroponico_logs/"
)

# Entrenamiento
model.learn(total_timesteps=200000)

# Guardar el modelo
model.save("sac_hidroponico_model")

# Guardar el normalizador de entorno (importante!)
env.save("hidroponico_vecnormalize.pkl")

print("Modelo entrenado y guardado exitosamente!")