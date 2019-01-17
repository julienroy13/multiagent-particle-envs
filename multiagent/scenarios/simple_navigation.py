import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, n_agents=5):
        world = World()
        # set any world properties first
        world.clip_positions = True
        world.dim_c = 0
        num_landmarks = 1
        num_agents = n_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.adversary = False
            agent.silent = True
            agent.collide = True
            agent.is_colliding = {other_agent.name:False for other_agent in world.agents if agent is not other_agent}
            agent.size = 0.05
            agent.accel = 0.5
            agent.max_speed = 0.25
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.54, 0.82, 0.98])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1., 1., world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1., 1., world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # def benchmark_data(self, agent, world):
    #     rew = 0
    #     collisions = 0
    #     occupied_landmarks = 0
    #     min_dists = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #         min_dists += min(dists)
    #         rew -= min(dists)
    #         if min(dists) < 0.1:
    #             occupied_landmarks += 1
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #                 collisions += 1
    #     return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        if agent1 is agent2 or not agent1.collide or not agent2.collide:
            return False

        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size

        if dist < dist_min:
            agent1.is_colliding[agent2.name] = True
            agent2.is_colliding[agent1.name] = True
            return True

        else:
            agent1.is_colliding[agent2.name] = False
            agent2.is_colliding[agent1.name] = False
            return False

    def count_collisions(self, agent, world):
        n_collisions = 0
        for a in world.agents:
            if self.is_collision(agent, a):
                n_collisions += 1

        # Sets agent's color based on whether it is colliding or not
        if any(agent.is_colliding.values()):
            agent.color = np.array([0.27, 0.44, 0.55])
        else:
            agent.color = np.array([0.54, 0.82, 0.98])

        return n_collisions


    def reward(self, agent, world):
        # Agents are rewarded based on the sum of all agent's distance to the landmark(s), and penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= np.sum(dists)

        if agent.collide:
            rew -= 1. * self.count_collisions(agent, world)

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # position of all other agents
        other_pos = []
        for other_agent in world.agents:
            if other_agent is agent:
                continue
            else:
                other_pos.append(other_agent.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
