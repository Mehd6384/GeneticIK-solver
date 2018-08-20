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



class Swarm(object): 

	def __init__(self, nb_ind, joints_length, config_ini,  c1 = 0.25, c2 = 0.35):

		self.nb_joints = config_ini.reshape(-1).shape[0]
		self.nb_ind = nb_ind
		self.joints_length = joints_length
		self.c1 = c1
		self.c2 = c2 
		
		self.pop = np.zeros((nb_ind, self.nb_joints))
		self.pop[:] = config_ini #np.random.uniform(0., np.pi*2., (nb_ind, nb_joints))
		self.vel = np.random.uniform(-0.1,0.1, (self.pop.shape))
		
		self.global_best_score = 0.
		self.global_best_position = config_ini.copy() 
		self.personal_bests = np.zeros((nb_ind))
		self.personal_bests_pos = self.pop.copy() 


	def get_next(self): 

		# max_ind = np.argmax(self.bests_scores)
		return self.pop[-1].copy().reshape(-1)

	def eval(self, target_pos): 

		effector_positions = compute_effector_position(self.pop, self.joints_length)
		score = 1./distance(effector_positions, target_pos)

		return score.reshape(-1)


	def update(self, target_pos): 

		score = self.eval(target_pos)
		# input(score)
		epoch_best_score = np.max(score)
		if epoch_best_score > self.global_best_score: 
			self.global_best_score = epoch_best_score
			self.global_best_position = self.pop[np.argmax(score)].copy()
		

		inds = [i for i in range(score.shape[0]) if score[i] > self.personal_bests[i]]
		self.personal_bests[inds] = score[inds]
		self.personal_bests_pos[inds] = self.pop[inds]


		self.update_vel()

		self.update_pos()

		return self.get_next()

	def update_pos(self): 

		self.pop = self.pop + self.vel.copy()

	def update_vel(self): 

		best_global = self.global_best_position.copy() 

		vec_local = self.personal_bests_pos - self.pop 
		vec_global = best_global - self.pop

		c1_influence = np.random.uniform(0.,1., (self.nb_ind, 1))*self.c1
		c2_influence = np.random.uniform(0.,1., (self.nb_ind, 1))*self.c2

		self.vel += (c1_influence*vec_local + c2_influence*vec_global)
		self.vel = np.clip(self.vel, -0.1, 0.1)

# s = Swarm(3, 5, 0.2)

# s.update(np.array([0.7,0.5]))



