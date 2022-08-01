

class StateDescriptor():
    def __init__(self, current_obs, next_obs, action, reward_dict, done, info):

        self.current_obs = current_obs
        self.next_obs = next_obs
        self.action = action
        self.reward_dict = reward_dict
        self.done = done
        self.info = info

        pass