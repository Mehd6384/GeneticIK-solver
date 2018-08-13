import numpy as np 
import arcade 
from GA import GeneticSolver

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

	def __init__(self, pos_ini = None): 

		self.pos = pos_ini if pos_ini != None else np.random.uniform(0.,1.,(2))

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

	def __init__(self, nb_joints, joints_length):

		
		self.nb_joints, self.joints_length = nb_joints, joints_length
		self.r1 = Robot(nb_joints, joints_length)
		self.r2 = Robot(nb_joints, joints_length)
		self.target = Target()

		self.solver = GeneticSolver(10, self.nb_joints-1, joints_length)
		self.solver.init_pop(self.r1.config.reshape(1,-1))

	def step(self): 

		r = np.random.uniform(-0.1,0.1,(self.nb_joints-1))
		self.r1.rotate(r)
		nex_conf = self.solver.evolve(self.target.pos)
		self.r1.set_config(nex_conf)

		d = self.compute_distance_end_effector_target()
		if d < 0.03: 
			self.reset()
		

	def reset(self): 

		self.r1 = Robot(self.nb_joints, self.joints_length)
		self.r2 = Robot(self.nb_joints, self.joints_length)
		self.target = Target()

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


		p = compute_effector_position(self.world.r1.config.reshape(1,-1), self.world.r1.joints_length)
		arcade.draw_text("Effector pos: {}\nDistance target: {}".format(p, distance(p, self.world.target.pos)), p[0]*self.size, p[1]*self.size, arcade.color.WHITE, 12)


	def update(self, value): 

		a = 0
		self.world.step()


world = World(nb_joints = 5, joints_length = 0.1)
render = Render(world)
# for i in range(2000): 

	# world.step(np.random.uniform(-0.1,0.1, (5)))
	# world.step()
arcade.run()