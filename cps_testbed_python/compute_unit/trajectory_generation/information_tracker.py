from dataclasses import dataclass


@dataclass
class TrajectoryContent:
	coefficients: any
	trajectory_start_time: any
	last_trajectory: any   # trajectory calculated in the last round (to improve performance)
	init_state: any
	current_state: any
	id: int
	trajectory_calculated_by: int


class InformationTracker:
	def __init__(self):
		self.__information_list = {}

	def add_information(self, key, content):
		if key not in self.__information_list.keys():
			self.__information_list[key] = Information(content)
		else:
			self.__information_list[key].set_content(content)

	def add_unique_information(self, key, content):
		if key not in self.__information_list.keys():
			self.__information_list[key] = Information(content)
		else:
			self.__information_list[key].set_unique_content(content)

	def set_deprecated(self, key):
		if key in self.__information_list.keys():
			self.__information_list[key].set_deprecated()

	@property
	def keys(self):
		return self.__information_list.keys()

	def get_information(self, key):
		return self.__information_list[key]

	def get_all_information(self):
		return self.__information_list

	@property
	def all_information_unique(self):
		unique = True
		for key in self.__information_list.keys():
			unique = unique and self.__information_list[key].is_unique
		return unique

	@property
	def no_information_deprecated(self):
		no_deprecated = True
		for key in self.__information_list.keys():
			no_deprecated = no_deprecated and not self.__information_list[key].is_deprecated
		return no_deprecated


class Information:
	def __init__(self, content):
		self.__deprecated = False
		self.__content = [content]
		self.__valid = True  # if data is valid, i.e. all data in this fulfills some constraints

	@property
	def content(self):
		# content is deprecated, return None.
		#if self.__deprecated:
		#	return None
		return self.__content

	@property
	def is_deprecated(self):
		return self.__deprecated

	@property
	def is_unique(self):
		return len(self.__content) == 1

	@property
	def valid(self):
		return self.__valid

	@valid.setter
	def valid(self, valid):
		self.__valid = valid

	def set_deprecated(self):
		self.__deprecated = True

	def set_content(self, content):
		"""
		set content to information. If using this method, the former content will be kept and the information will
		become uncertain (it is not clear, which content is true)
		"""
		if self.__deprecated:
			#self.__deprecated = False
			#self.__content = [content]
			pass
		self.__content.append(content)

	def set_unique_content(self, content):
		"""
		set content to information. If using this method, the former concent will be withdrawn and the parameter will
		be set as the uniwue content (it is clear, that content is the only true content.)
		"""
		self.__deprecated = False
		self.__content = [content]
		self.__valid = True
