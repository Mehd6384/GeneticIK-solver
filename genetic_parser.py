from argparse import ArgumentParser 

class Parser(ArgumentParser): 

	def __init__(self): 

		ArgumentParser.__init__(self)

		self.add_argument("--algo", help = "Algo to use",default = "ga") 
		self.add_argument("--pop_size", help = "Population size", type = int, default = 15)
		self.add_argument("--elite_pop", help = "Elite population size", type = int, default = 3)
		self.add_argument("--nb_joints", help = "Nb joints", type = int, default = 4)
		self.add_argument("--joint_length", help = "Joint length", type = float, default = 0.15)
		self.add_argument("--nb_obstacles", help = "Nb obstacles", type = int, default = 0)
		self.add_argument("--max_steps", help = "Max steps", type = int, default = 250)
		self.add_argument("--mode", help = "Run mode", default = "show")
		self.add_argument("--nb_eval", help = "Nb evaluations in eval mode", type = int, default = 1000)


	@property
	def algo(self):
		return self.parse_args().algo

	@property
	def pop_size(self):
		return self.parse_args().pop_size

	@property
	def elite_pop(self):
		return self.parse_args().elite_pop

	@property
	def nb_joints(self):
		return self.parse_args().nb_joints

	@property
	def joint_length(self):
		return self.parse_args().joint_length

	@property
	def max_steps(self):
		return self.parse_args().max_steps
	
	@property
	def mode(self):
		return self.parse_args().mode

	@property
	def nb_eval(self):
		return self.parse_args().nb_eval

	@property
	def nb_obstacles(self):
		return self.parse_args().nb_obstacles


