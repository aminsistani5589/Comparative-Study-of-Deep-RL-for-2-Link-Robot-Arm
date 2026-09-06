# make_gif.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from SARSA import TwoLinkArm, DeepSARSAAgent

def draw_arm_frame(state, goal, L1=1.0, L2=1.0):
    """رسم یک فریم با هدف ثابت"""
    th1 = np.arctan2(state[1], state[0])
    th2 = np.arctan2(state[3], state[2])
    goal_x, goal_y = goal

    ex = L1 * np.cos(th1)
    ey = L1 * np.sin(th1)
    eex = ex + L2 * np.cos(th1 + th2)
    eey = ey + L2 * np.sin(th1 + th2)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.plot([0, ex], [0, ey], color='blue', linewidth=4)
    ax.plot([ex, eex], [ey, eey], color='red', linewidth=4)
    ax.scatter(0, 0, color='black', s=100)
    ax.scatter(ex, ey, color='black', s=60)
    ax.scatter(eex, eey, color='green', s=80)
    ax.scatter(goal_x, goal_y, color='orange', marker='*', s=200)
    ax.set_title(f'θ₁={np.degrees(th1):.1f}°, θ₂={np.degrees(th2):.1f}°')

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def create_gif(
    actor_weights_path='sarsa_deep_actor.pth',
    gif_path='arm_animation.gif',
    max_steps=1000,          # حداکثر قدم‌های مجاز
    fps=20,                  # فریم بر ثانیه
    duplicate_frames=1,      # هر فریم چند بار تکرار شود (برای حرکت آهسته‌تر)
):
    env = TwoLinkArm()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DeepSARSAAgent(state_dim, action_dim)
    agent.actor.load_state_dict(torch.load(actor_weights_path, map_location='cpu'))
    agent.actor.eval()

    state, _ = env.reset()
    goal = (state[6], state[7])   # هدف ثابت از ابتدای اپیزود

    frames = []
    for step in range(max_steps):
        # رسم فریم فعلی
        frame = draw_arm_frame(state, goal)
        # اضافه کردن فریم‌های تکراری برای کندتر شدن انیمیشن
        for _ in range(duplicate_frames):
            frames.append(frame)

        # انتخاب عمل و اعمال به محیط
        action, _ = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state

        if terminated or truncated:
            # فریم آخر هم چند بار تکرار شود تا پایان بهتر دیده شود
            frame = draw_arm_frame(state, goal)
            for _ in range(duplicate_frames * 5):  # ۵ برابر تکرار برای مکث
                frames.append(frame)
            break

    # ذخیره گیف
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"GIF saved to {gif_path} with {len(frames)} frames.")

if __name__ == '__main__':
    create_gif(
        actor_weights_path='sarsa_deep_actor.pth',
        gif_path='arm_animation.gif',
        max_steps=1000,
        fps=15,               # کمتر = کندتر
        duplicate_frames=2,   # هر فریم دو بار = حرکت نرم‌تر
    )