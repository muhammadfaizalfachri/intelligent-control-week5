import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# Inisialisasi environment (CartPole dari Gymnasium)
env = gym.make("CartPole-v1")

# Parameter DRL
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Bangun model Deep Q-Network (DQN)
model = keras.Sequential([
    keras.layers.Input(shape=(state_size,)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(action_size, activation="linear")
])
model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

# Fungsi pilih aksi (epsilon-greedy)
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)  # eksplorasi
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])  # eksploitasi

# Proses training
episodes = 500
rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(1000):
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            # Cetak hanya reward dan epsilon
            print(f"Episode {episode+1}/{episodes} | Reward: {total_reward:.0f} | Epsilon: {epsilon:.4f}")
            break

    rewards_per_episode.append(total_reward)

    # Update model (experience replay)
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)

        for s, a, r, ns, d in minibatch:
            target = r
            if not d:
                target += gamma * np.amax(model.predict(ns, verbose=0)[0])
            target_f = model.predict(s, verbose=0)
            target_f[0][a] = target
            model.fit(s, target_f, epochs=1, verbose=0)

    # Update epsilon (exploration decay)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

model.save("dqn_cartpole.keras")
print("Training selesai!")

# === Plot grafik reward per episode ===
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Perkembangan Reward Selama Training DQN")
plt.show()
