import numpy as np 


def compute_effector_position(config, joints_length): 

	j_current = np.array([0.5,0.5])
	for i in range(config.shape[1]): 
		c = config[:,i].reshape(-1,1)
		angle_mat = np.concatenate([np.cos(c), np.sin(c)],1)
		j_current = j_current + joints_length*angle_mat

	return j_current

def distance(p1, p2): 

	return np.sqrt(np.sum(np.power(p1-p2,2),1))

def compute_forward_kinematics(genes, joints_length): 

	joints_positions = np.zeros((genes.shape[0], 2, genes.shape[1])) 

	for j in range(joints_positions.shape[2]):

		x = np.cos(genes[:,j])*joints_length
		y = np.sin(genes[:,j])*joints_length
		current_joint_pos = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], 1)

		if j > 0: 

			current_joint_pos += joints_positions[:,:,j-1]


		joints_positions[:,:,j] = current_joint_pos

	joints_positions += 0.5

	return joints_positions


def distance_to_obstacles(genes, obstacles, joint_length): 

	
	joints_positions = compute_forward_kinematics(genes, joint_length)
	distances = np.zeros((genes.shape[0], joints_positions.shape[2], len(obstacles)))
	for joint in range(distances.shape[1]): 
		for ob in range(len(obstacles)): 

			current_joint = joints_positions[:,:,joint]
			current_distances = distance(current_joint, obstacles[ob].pos)

			distances[:,joint,ob] = current_distances

	mean_distances_for_each_joint = np.mean(distances, 2)
	mean_distance_pop = np.mean(mean_distances_for_each_joint, 1).reshape(-1,1)

	return mean_distance_pop

def compute_link_collision(genes, obstacles, joint_length): 

	joints_positions = compute_forward_kinematics(genes, joint_length)
	# Joint poisition has shape (pop, 2, nb_joints)

	point_line_distances = np.zeros((joints_positions.shape[0],joints_positions.shape[2] -1, len(obstacles)))

	for o in range(point_line_distances.shape[2]): 
		for j in range(point_line_distances.shape[1]): 

			p1 = joints_positions[:,:,j]
			p2 = joints_positions[:,:,j+1]

			obs_pos = obstacles[o].pos.copy()


			a_term = p1[:,1] - p2[:,1]*obs_pos[0]
			b_term = p1[:,0] - p2[:,0]*obs_pos[0]
			c_term = p1[:,1]*p2[:,0] - p1[:,0]*p2[:,1]

			num = np.abs(a_term - b_term + c_term)
			den = np.sqrt(np.square(p1[:,0] - p2[:,0]) + np.square(p1[:,1] - p2[:,1]))

			current_distances = num/den

			point_line_distances[:, j, o] = current_distances.copy()


	return point_line_distances
	
class GeneticSolver: 

	def __init__(self, population_size, joint_length, example):

		self.population_size = population_size
		self.nb_joints = example.reshape(1,-1).shape[1]
		self.joint_length = joint_length

		self.init_pop(example)

	def init_pop(self, example, noise = 0.01): 

		self.genes = np.tile(example, [self.population_size,1])
		self.genes += np.random.uniform(-noise, noise, self.genes.shape)

	def compute_success_end_effector_target(self, target_pos): 

		target = np.ones((self.population_size,2))*target_pos
		end_effector_positions = compute_effector_position(self.genes, self.joint_length)
		error = distance(target, end_effector_positions)
		
		return error 

	def compute_success_obstacles(self, obstacles): 

		dis = distance_to_obstacles(self.genes, obstacles, self.joint_length)
		link_collision = compute_link_collision(self.genes, obstacles, self.joint_length)

		link_collision[link_collision < 0.03] = 0.001
		link_collision = np.mean(np.mean(link_collision, 1), 1)

		return 1./link_collision

	def evolve(self, target_pos, obstacles = None): 


		error = self.compute_success_end_effector_target(target_pos)
		if obstacles != None: 

			error_obstacles = self.compute_success_obstacles(obstacles).reshape(-1)
		

			error += error_obstacles

		ordered_bests = np.argsort(error.reshape(-1))

		best_ind = self.genes[ordered_bests[0], :].copy()

		p1, p2 = self.genes[ordered_bests[0],:], self.genes[ordered_bests[1],:]
		for p in range(self.population_size): 
			if p not in ordered_bests[0:2]:
				parent = p1.copy() if np.random.random() < 0.5 else p2.copy()
				self.genes[p,:] = parent + np.random.uniform(-0.1,0.1, (parent.shape))

		return best_ind


	def get_genes(self):
		return self.genes.copy()
	
	def __repr__(self): 

		return 'GeneticSolver for {} parameters with pop_size: {}'.format(self.nb_joints, self.population_size) 


class GeneticSolverWithElitism(GeneticSolver): 

	def __init__(self, population_size, joint_length, example, nb_elites): 

		GeneticSolver.__init__(self, population_size, joint_length, example)
		self.nb_elites = nb_elites

	def evolve(self, target_pos):


		error = self.compute_success_end_effector_target(target_pos)

		ordered_bests = np.argsort(error.reshape(-1))


		best_ind = self.genes[ordered_bests[0], :].copy()
		new_pop = np.zeros_like(self.genes) 
		elites = self.genes[ordered_bests[0:self.nb_elites], :]

		new_pop[0:self.nb_elites] = elites
		for i in range(self.nb_elites, new_pop.shape[0]): 
			ind_parents = np.random.randint(0, elites.shape[0], (2)) # selecting parents

			# creating new individual from mating 
			new_individual = np.array([elites[ind_parents[0],i] if np.random.random() < 0.5 
				else elites[ind_parents[1],i] for i in range(elites.shape[1])]).reshape(1,-1)


			# applying mutation 
			if np.random.random() < 0.2: 
				# print('Applying mutation')
				new_individual += np.random.normal(0,0.05, new_individual.shape)

			new_pop[i] = new_individual


			# input(new_pop)
			# print('Elites\n {}\nIndices parents: {}\nParents: {}'.format(elites, ind_parents, elites[ind_parents]))

			# print('New ind: {}'.format(new_individual))
			# input()
			# input(new_individual)

		self.genes = new_pop.copy()

		return best_ind

	def __repr__(self): 

		return 'GeneticSolver for {} parameters with pop_size: {} and elite size: {}'.format(self.nb_joints, self.population_size, self.nb_elites) 

