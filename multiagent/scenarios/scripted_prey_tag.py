# CODE TAKEN FROM multiagent.scenarios.simple_tag.py
import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
from multiagent.policy import RunnerPolicy


class Scenario(BaseScenario):
    def make_world(self, n_preds=2, n_preys=1):  # leave those default values to be able to load models trained on this script that precede 08/01/2019
        world = World()
        # set any world properties first
        world.clip_positions = True
        world.dim_c = 0
        num_good_agents = n_preys
        num_adversaries = n_preds
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0  # No landmark to avoid lucky catches since the prey is scripted and cannot avoid them
        # add policy for always_scripted agents
        self.runner_policy = RunnerPolicy()
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.05 if agent.adversary else 0.04
            agent.accel = 1. if agent.adversary else 1.5
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1. if agent.adversary else 1.5
            agent.always_scripted = True if not agent.adversary else False
            if not agent.adversary and agent.always_scripted:
                agent.action_callback = self.runner_policy.action
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        while True:
            # set random initial states
            for agent in world.agents:
                if not(agent.adversary):
                    agent.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
                else:
                    agent.state.p_pos = np.random.uniform(-1., +1., world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            for i, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)
            # checks for overlaps between agent's initial positions
            overlap = 0
            for agent_i in world.agents:  # agent_i is an prey
                if agent_i.adversary:
                    continue
                for agent_j in world.agents:  # agent_j is a predator
                    if not(agent_j.adversary):
                        continue
                    if agent_i is agent_j:
                        continue
                    if np.sqrt(np.sum(np.square(agent_i.state.p_pos - agent_j.state.p_pos))) < 1.5*(agent_i.size + agent_j.size):
                        overlap += 1
            # if there is a single overlap, we re-do the position initialization
            if overlap > 0:
                continue
            else:
                break


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # def bound(x):
        #     if x < 0.9:
        #         return 0
        #     if x < 1.0:
        #         return (x - 0.9) * 10
        #     return min(np.exp(2 * x - 2), 10)
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # get relative positions to the walls
        wall_pos = []
        for wall in world.walls:
            if wall.orient == 'H':
                wall_pos.append(agent.state.p_pos[1] - wall.axis_pos)
            elif wall.orient == 'V':
                wall_pos.append(agent.state.p_pos[0] - wall.axis_pos)
        wall_pos = [np.array(wall_pos)]
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + wall_pos)
