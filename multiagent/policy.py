import numpy as np
from pyglet.window import key
from multiagent.core import Action

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, *args):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, viewer_id):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[viewer_id].window.on_key_press = self.key_press
        env.viewers[viewer_id].window.on_key_release = self.key_release

    def action(self, *args):
        """
        Arguments are ignored for InteractivePolicy. However, the call to action_callback(agent, self)
        in multiagent.core.py requires that this callback method accepts an agent and world if provided.
        """
        action = Action()
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(2) # 5-d because of no-move action
            if self.move[0]: u[0] += 0.4
            if self.move[1]: u[0] -= 0.4
            if self.move[3]: u[1] += 0.4
            if self.move[2]: u[1] -= 0.4
        action.u = u
        return action

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.RIGHT:  self.move[0] = True
        if k==key.LEFT: self.move[1] = True
        if k==key.DOWN:    self.move[2] = True
        if k==key.UP:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.RIGHT:  self.move[0] = False
        if k==key.LEFT: self.move[1] = False
        if k==key.DOWN:    self.move[2] = False
        if k==key.UP:  self.move[3] = False
