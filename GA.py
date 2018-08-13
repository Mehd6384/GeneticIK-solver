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

class GeneticSolver: 

	def __init__(self, population_size, nb_joints, joint_length):

		self.population_size = population_size
		self.nb_joints = nb_joints
		self.joint_length = joint_length

	def init_pop(self, initial_config, noise = 0.01): 

		self.genes = np.random.uniform(-noise, noise, (self.population_size, self.nb_joints))
		self.genes += initial_config

	def evolve(self, target_pos): 

		target = np.ones((self.population_size,2))*target_pos
		end_effector_positions = compute_effector_position(self.genes, self.joint_length)
		error = distance(target, end_effector_positions)

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