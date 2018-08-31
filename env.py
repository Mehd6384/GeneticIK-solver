import numpy as np 
import arcade 
import GA 
from pso import Swarm
from genetic_parser import Parser 

import matplotlib.pyplot as plt 
plt.style.use('dark_background')


def compute_effector_position(config, joints_length): 

	j_current = np.array([0.5,0.5])
	config = config.reshape(1,-1)
	for i in range(config.shape[1]): 
		c = config[:,i].reshape(-1)
		angle_mat = np.hstack([np.cos(c), np.sin(c)])
		j_current = j_current +joints_length* angle_mat

	return j_current

def distance(p1, p2): 
	return np.sqrt(np.sum(np.power(p1-p2,2)))

def normalize_angle(angle): 
	return angle%(np.pi*2.)

class Target: 

	def __init__(self, max_distance): 

		angle = np.random.uniform(0, np.pi*2.)
		self.pos = np.array([np.cos(angle),np.sin(angle)])*max_distance*np.random.random()
		self.pos += 0.5

	@property
	def draw_info(self):
		return self.pos.copy()

class Robot: 

	def __init__(self, nb_joints, joints_length, config_ini = None): 

		self.nb_joints = nb_joints-1 
		self.joints_length = joints_length 
		self.config = np.random.uniform(0.,np.pi*2, (nb_joints-1)) # last joint doesn't rotate
		
	def set_config(self, config): 

		self.config = config.copy()

	def rotate(self, vec): 
		config = self.config.copy()
		# print(config)
		config += vec 
		# print(config)
		# input()
		self.config = config.copy()

	@property
	def draw_info(self): 

		positions = np.zeros((self.nb_joints+1, 2))
		positions[0] = [0.5,0.5]

		for i in range(1,positions.shape[0]):

			x = positions[i-1,0] + self.joints_length*np.cos(self.config[i-1])
			y = positions[i-1,1] + self.joints_length*np.sin(self.config[i-1])

			positions[i,:] = [x,y]

		lines = np.zeros((self.nb_joints, 4))
		for i in range(lines.shape[0]): 
			x = positions[i:i+2,:]
			lines[i,:] = positions[i:i+2,:].reshape(-1)
		return positions, lines


class World: 

	def __init__(self, args):

		self.max_steps = args.max_steps
		self.steps = 0
		
		self.nb_joints, self.joints_length = args.nb_joints, args.joint_length
		self.max_distance = (self.nb_joints-1)*self.joints_length
		self.nb_elites = args.elite_pop

		self.r1 = Robot(self.nb_joints, self.joints_length)
		self.target = Target(self.max_distance)

		self.nb_obstacles = args.nb_obstacles
		self.obstacles = None if self.nb_obstacles == 0 else [Target(self.max_distance) for i in range(self.nb_obstacles)]

		if args.algo == 'ga': 
			self.solver = GA.GeneticSolver(args.pop_size, self.joints_length, self.r1.config)
		elif args.algo == 'ga_elite': 
			self.solver = GA.GeneticSolverWithElitism(args.pop_size, self.joints_length, self.r1.config, self.nb_elites)


	def step(self): 

		complete = False
		self.steps += 1

		if self.steps >= self.max_steps: 
			complete = True
		
		nex_conf = self.solver.evolve(self.target.pos, self.obstacles)
		self.r1.set_config(nex_conf)

		d = self.compute_distance_end_effector_target()
		if d < 0.03: 
			complete = True


		return complete, self.steps
		

	def reset(self): 

		# self.r1 = Robot(self.nb_joints, self.joints_length)
		self.steps = 0
		self.target = Target(self.max_distance)

	def compute_distance_end_effector_target(self): 

		return distance(compute_effector_position(self.r1.config, self.joints_length), self.target.pos)


class Render(arcade.Window): 

	def __init__(self, world, size = 700): 

		arcade.Window.__init__(self, size, size, "Gen")
		self.size = size
		self.world = world 

	def on_draw(self): 	

		arcade.start_render()

		pos,lines = self.world.r1.draw_info 
		for l in lines: 
			scaled_l = l.copy()*self.size
			arcade.draw_line(*scaled_l, (215, 101, 0), 10)
		for p in pos: 
			scaled_p = p.copy()*self.size
			arcade.draw_circle_filled(scaled_p[0], scaled_p[1], 10, (19, 131, 235))

		target_pos = self.world.target.draw_info*self.size
		arcade.draw_circle_filled(target_pos[0], target_pos[1], 15, (250, 15, 0))
		arcade.draw_circle_filled(target_pos[0], target_pos[1], 10, (250, 250, 250))
		arcade.draw_circle_filled(target_pos[0], target_pos[1], 5, (250, 15, 0))


		if self.world.obstacles != None: 
			for o in self.world.obstacles: 

				ob_pos = o.draw_info*self.size
				arcade.draw_circle_filled(ob_pos[0], ob_pos[1], 15, (250, 250, 0))
				arcade.draw_circle_filled(ob_pos[0], ob_pos[1], 10, (0, 0, 0))
				arcade.draw_circle_filled(ob_pos[0], ob_pos[1], 5, (250, 250, 0))


		p = compute_effector_position(self.world.r1.config.reshape(1,-1), self.world.r1.joints_length)
		arcade.draw_text("Effector pos: {}\nDistance target: {}".format(p, distance(p, self.world.target.pos)), p[0]*self.size, p[1]*self.size, arcade.color.WHITE, 12)


	def update(self, value): 

		a = 0
		over, _ = self.world.step()
		if over: self.world.reset()


args = Parser()

if args.mode == 'show': 

	world = World(args)
	render = Render(world)
	# for i in range(2000): 

		# world.step(np.random.uniform(-0.1,0.1, (5)))
		# world.step()
	arcade.run()

elif args.mode == 'eval': 

	world = World(args)

	epochs = args.nb_eval
	recap = []
	for epoch in range(epochs): 

		over = False 
		while not over: 

			over, steps = world.step()
			if over: 
				recap.append(steps)

		if epoch%100 == 0: 
			print('Evaluating... {}/{}'.format(epoch, epochs))

	plt.hist(recap, bins = 4, rwidth = 0.8)
	plt.show()

