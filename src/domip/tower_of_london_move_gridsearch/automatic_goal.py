from typing import Union

import torch

from domip.base_classes.goals import Goal

# Color mappings: 0=empty, 1=RED, 2=BLUE, 3=YELLOW
# Slots: [0=P0-H0, 1=P1-H0, 2=P1-H1, 3=P2-H0, 4=P2-H1, 5=P2-H2]


class Pos11Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos11Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[3, 2] - round(likelihood_slots[3, 2].item()) * likelihood_slots[4, 1] - round(likelihood_slots[3, 2].item()) * round(likelihood_slots[4, 1].item()) * likelihood_slots[5, 0]
			return cost, cost
		return goal_func


class Pos21Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos21Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[3, 2] - round(likelihood_slots[3, 2].item()) * likelihood_slots[4, 0] - round(likelihood_slots[3, 2].item()) * round(likelihood_slots[4, 0].item()) * likelihood_slots[5, 1]
			return cost, cost
		return goal_func


class Pos31Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos31Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[3, 0] - round(likelihood_slots[3, 0].item()) * likelihood_slots[4, 2] - round(likelihood_slots[3, 0].item()) * round(likelihood_slots[4, 2].item()) * likelihood_slots[5, 1] 
			return cost, cost
		return goal_func


class Pos41Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos41Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[3, 0] - round(likelihood_slots[3, 0].item()) * likelihood_slots[4, 1] - round(likelihood_slots[3, 0].item()) * round(likelihood_slots[4, 1].item()) * likelihood_slots[5, 2]
			return cost, cost
		return goal_func


class Pos51Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos51Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[3, 1] - round(likelihood_slots[3, 1].item()) * likelihood_slots[4, 0] - round(likelihood_slots[3, 1].item()) * round(likelihood_slots[4, 0].item()) * likelihood_slots[5, 2]
			return cost, cost
		return goal_func


class Pos61Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos61Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[3, 1] - round(likelihood_slots[3, 1].item()) * likelihood_slots[4, 2] - round(likelihood_slots[3, 1].item()) * round(likelihood_slots[4, 2].item()) * likelihood_slots[5, 0]
			return cost, cost
		return goal_func


class Pos12Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos12Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 0] - likelihood_slots[3, 2] - round(likelihood_slots[3, 2].item()) * likelihood_slots[4, 1]
			return cost, cost
		return goal_func


class Pos22Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos22Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 1] - likelihood_slots[3, 2] - round(likelihood_slots[3, 2].item()) * likelihood_slots[4, 0]
			return cost, cost
		return goal_func


class Pos32Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos32Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 1] - likelihood_slots[3, 0] - round(likelihood_slots[3, 0].item()) * likelihood_slots[4, 2]
			return cost, cost
		return goal_func


class Pos42Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos42Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 2] - likelihood_slots[3, 0] - round(likelihood_slots[3, 0].item()) * likelihood_slots[4, 1]
			return cost, cost
		return goal_func


class Pos52Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos52Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 2] - likelihood_slots[3, 1] - round(likelihood_slots[3, 1].item()) * likelihood_slots[4, 0]
			return cost, cost
		return goal_func


class Pos62Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos62Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 0] - likelihood_slots[3, 1] - round(likelihood_slots[3, 1].item()) * likelihood_slots[4, 2]
			return cost, cost
		return goal_func


class Pos13Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos13Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 0] - likelihood_slots[3, 2] - round(likelihood_slots[3, 2].item()) *likelihood_slots[4, 1]
			return cost, cost
		return goal_func


class Pos23Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos23Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 1] - likelihood_slots[3, 2] - round(likelihood_slots[3, 2].item()) * likelihood_slots[4, 0]
			return cost, cost
		return goal_func


class Pos33Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos33Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 1] - likelihood_slots[3, 0] - round(likelihood_slots[3, 0].item()) * likelihood_slots[4, 2]
			return cost, cost
		return goal_func


class Pos43Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos43Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 2] - likelihood_slots[3, 0] - round(likelihood_slots[3, 0].item()) * likelihood_slots[4, 1]
			return cost, cost
		return goal_func


class Pos53Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos53Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 2] - likelihood_slots[3, 1] - round(likelihood_slots[3, 1].item()) * likelihood_slots[4, 0]
			return cost, cost
		return goal_func


class Pos63Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos63Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 0] - likelihood_slots[3, 1] - round(likelihood_slots[3, 1].item()) * likelihood_slots[4, 2]
			return cost, cost
		return goal_func


class Pos14Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos14Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 0] - round(likelihood_slots[1, 0].item()) * likelihood_slots[2, 1] - likelihood_slots[3, 2]
			return cost, cost
		return goal_func


class Pos24Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos24Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 1] - round(likelihood_slots[1, 1].item()) * likelihood_slots[2, 0] - likelihood_slots[3, 2]
			return cost, cost
		return goal_func


class Pos34Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos34Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 1] - round(likelihood_slots[1, 1].item()) * likelihood_slots[2, 2] - likelihood_slots[3, 0]
			return cost, cost
		return goal_func


class Pos44Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos44Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 2] - round(likelihood_slots[1, 2].item()) * likelihood_slots[2, 1] - likelihood_slots[3, 0]
			return cost, cost
		return goal_func


class Pos54Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos54Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 2] - round(likelihood_slots[1, 2].item()) * likelihood_slots[2, 0] - likelihood_slots[3, 1]
			return cost, cost
		return goal_func


class Pos64Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos64Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[1, 0] - round(likelihood_slots[1, 0].item()) * likelihood_slots[2, 2] - likelihood_slots[3, 1]
			return cost, cost
		return goal_func


class Pos15Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos15Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 1] - likelihood_slots[1, 0] - likelihood_slots[3, 2]
			return cost, cost
		return goal_func


class Pos25Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos25Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 0] - likelihood_slots[1, 1] - likelihood_slots[3, 2]
			return cost, cost
		return goal_func


class Pos35Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos35Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 2] - likelihood_slots[1, 1] - likelihood_slots[3, 0]
			return cost, cost
		return goal_func


class Pos45Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos45Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 1] - likelihood_slots[1, 2] - likelihood_slots[3, 0]
			return cost, cost
		return goal_func


class Pos55Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos55Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 0] - likelihood_slots[1, 2] - likelihood_slots[3, 1]
			return cost, cost
		return goal_func


class Pos65Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos65Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 2] - likelihood_slots[1, 0] - likelihood_slots[3, 1]
			return cost, cost
		return goal_func


class Pos16Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos16Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 1] - likelihood_slots[1, 0] - round(likelihood_slots[1, 0].item()) *likelihood_slots[2, 2]
			return cost, cost
		return goal_func


class Pos26Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos26Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 0] - likelihood_slots[1, 1] - round(likelihood_slots[1, 1].item()) * likelihood_slots[2, 2]
			return cost, cost
		return goal_func


class Pos36Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos36Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 2] - likelihood_slots[1, 1] - round(likelihood_slots[1, 1].item()) * likelihood_slots[2, 0]
			return cost, cost
		return goal_func


class Pos46Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos46Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 1] - likelihood_slots[1, 2] - round(likelihood_slots[1, 2].item()) * likelihood_slots[2, 0]
			return cost, cost
		return goal_func


class Pos56Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos56Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 0] - likelihood_slots[1, 2] - round(likelihood_slots[1, 2].item()) * likelihood_slots[2, 1]
			return cost, cost
		return goal_func


class Pos66Goal(Goal):

	def __init__(self, is_active:bool, dtype:Union[torch.dtype, None] = None, device : Union[torch.device, None] = None, mockbuild : bool = False):

		super().__init__(name="Pos66Goal", is_active=is_active, mockbuild=False, fulfilled_value=0.1)

	def define_goal_cost_function(self):

		def goal_func(likelihood_slots):
			cost = 3.0 - likelihood_slots[0, 2] - likelihood_slots[1, 0] - round(likelihood_slots[1, 0].item()) * likelihood_slots[2, 1]
			return cost, cost
		return goal_func
