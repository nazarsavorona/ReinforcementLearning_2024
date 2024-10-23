import gym
import time
from main import Policy
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

torch.manual_seed(0) # set random seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dict = torch.load('checkpoint_best.pth')

policy = Policy()
policy.load_state_dict(state_dict)
policy = policy.to(device)

def show_smart_agent():
    env = gym.make('Acrobot-v1')
    recorder = VideoRecorder(env, path='./video.mp4', enabled=True)
    state = env.reset()

    score = 0

    for t in range(1000):
        recorder.capture_frame()
        action, _ = policy.act(state)
        # actions = [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # action = actions[t % actions.__len__()]
        print(action)
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward

        if done:
            break
        time.sleep(0.05)
    print(f'Score: {score}')

    env.close()


if __name__ == '__main__':
    show_smart_agent()