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
            u = np.zeros(2)
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


class RunnerPolicy(Policy):
    """
    Policy for prey in simple_tag setups
    Simply runs away from the adversaries and the limits of the environment.
    Driven by repulsive forces inversely proportional to its distance with those entities.
    Only creates movement action, not communication.
    """
    def __init__(self, max_force=1.):
        self.max_force = max_force
        super(RunnerPolicy, self).__init__()

    def action(self, agent, world):
        action = Action()
        force = np.zeros(2)

        # Forces from predator agents
        for other_agent in world.agents:
            if other_agent.adversary and agent is not other_agent:
                force_vec = agent.state.p_pos - other_agent.state.p_pos
                force_norm = np.sqrt(np.sum(np.square(force_vec)))
                force += force_vec / force_norm**2 if force_norm**2 > 0.001 else force_vec / 0.001

        # Forces from environment limits
        d_right = agent.state.p_pos[0] - 1.
        d_left = agent.state.p_pos[0] + 1.
        d_up = agent.state.p_pos[1] - 1.
        d_down = agent.state.p_pos[1] + 1.
        force[0] +=  d_right / d_right**2 if d_right**2 > 0.001 else d_right / 0.001
        force[0] +=  d_left / d_left**2 if d_left**2 > 0.001 else d_left / 0.001
        force[1] +=  d_up / d_up**2 if d_up**2 > 0.001 else d_up / 0.001
        force[1] +=  d_down / d_down**2 if d_down**2 > 0.001 else d_down / 0.001

        force = force if np.sqrt(np.sum(force**2)) < self.max_force \
            else self.max_force * force / np.sqrt(np.sum(force**2))
        action.u = force
        return action


class RusherPolicy(Policy):
    """
    Policy for predators in simple_tag setups
    Simply rushes towards the prey(s).
    Driven by attractive forces proportional to its distance with the prey(s).
    Only creates movement action, not communication.
    """
    def __init__(self, max_force=1.):
        self.max_force = max_force
        super(RusherPolicy, self).__init__()

    def action(self, agent, world):
        action = Action()
        force = np.zeros(2)

        # Forces from prey agents
        for other_agent in world.agents:
            if not other_agent.adversary and agent is not other_agent:
                force_vec = other_agent.state.p_pos - agent.state.p_pos
                force_norm = np.sqrt(np.sum(force_vec**2))
                force += force_vec / force_norm**2 if force_norm**2 > 0.001 else force_vec / 0.001

        force = force if np.sqrt(np.sum(force**2)) < self.max_force \
            else self.max_force * force / np.sqrt(np.sum(force**2))
        action.u = force
        return action
