import json


class Data:
    def __init__(self, task_idx, initial_object_dict, goal):
        self.goal_idx = None
        self.task_idx = task_idx
        self.initial_object_dict = initial_object_dict
        self.current_object_dict = None
        self._goal = goal
        self.action_list = list()
        self._meaning = None
        self.utterance = None
        self._possible_solution = None
        self._objects_in_utterance = list()
        self._objects_in_meaning = list()
        self._useful_objects = list()
        self.level = None
        self._rsa_meaning = None
        self._objects_in_rsa_meaning = list()

        self._subgoal = None
        self.demo_actions = list()

    def append_action(self, action_name, arguments):
        self.action_list.append({'name': action_name, 'arguments': arguments})

    def set_utterance(self, utterance):
        self.utterance = utterance

    def set_private(self, meaning, possible_solution, rsa_meaning):
        self._meaning = meaning
        self._possible_solution = possible_solution
        self._rsa_meaning = rsa_meaning

    def set_answer_objects(self, oiu, oim, uo, oirm):
        self._objects_in_utterance = oiu
        self._objects_in_meaning = oim
        self._useful_objects = uo
        self._objects_in_rsa_meaning = oirm

    def get_meaning(self):
        return self._meaning

    def get_goal(self):
        return self._goal

    def get_possible_solution(self):
        return self._possible_solution

    def get_objects_in_meaning(self):
        return self._objects_in_meaning


def generate_json(data, quest_type, root='./'):
    if data.goal_idx is not None:
        goal = data.goal_idx
    else:
        goal = ''
    json_file = root + "json_files_{}/task_{}_{}_{}({}).json".format(quest_type, quest_type, goal, data.task_idx, data.level) # todo
    with open(json_file, "w+") as f:
        json.dump(data.__dict__, f)
    return


def load_from_json(file):
    with open(file, 'r') as f:
        json_str = json.load(f)
    task_idx = json_str['task_idx']
    initial_object_dict = json_str['initial_object_dict']
    goal = json_str['_goal']
    data = Data(task_idx, initial_object_dict, goal)
    data.action_list = json_str['action_list']
    data.current_object_dict = json_str['current_object_dict']
    data.set_utterance(json_str['utterance'])
    data.set_private(json_str['_meaning'], json_str['_possible_solution'], json_str['_rsa_meaning'])
    data.set_answer_objects(json_str['_objects_in_utterance'], json_str['_objects_in_meaning'], json_str['_objects_in_rsa_meaning'])
    return data
