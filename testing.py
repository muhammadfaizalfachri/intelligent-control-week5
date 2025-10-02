import gymnasium as gym
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# === Load environment ===
env = gym.make("CartPole-v1", render_mode="human")  # tampilkan animasi saat testing

# === Load model (tanpa compile agar tidak error) ===
model = keras.models.load_model("dqn_cartpole.keras")

# === Fungsi pilih aksi terbaik (tanpa eksplorasi) ===
def select_action(state):
    q_values = model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# === Pengujian ===
test_episodes = 10
rewards_per_test = []

for episode in range(test_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    total_reward = 0

    for t in range(500):  # batas langkah per episode
        action = select_action(state)  # selalu eksploitasi
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        total_reward += reward

        if done:
            print(f"[TEST] Episode {episode+1}/{test_episodes} | Reward: {total_reward}")
            break

    rewards_per_test.append(total_reward)

env.close()

# === Plot hasil reward selama testing ===
plt.plot(rewards_per_test, marker="o")
plt.xlabel("Test Episode")
plt.ylabel("Total Reward")
plt.title("Hasil Pengujian DQN pada CartPole")
plt.show()
