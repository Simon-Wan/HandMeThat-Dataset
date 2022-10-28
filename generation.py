import jacinle
import os.path as osp
import pdsketch as pds

import argparse
import multiprocessing as mp
import itertools

from utils import *
from sampling.scene_sampler import hierarchy2json, get_valid_positions, sample_scene
from sampling.goal_sampler import generate_one_goal
from planning.trajectory_generator import solve_goal
from data_processing.data_class import Data


def sample_and_define_scene(nodes, hierarchy, domain, valid_positions):
    object_dict, object_names = sample_scene(hierarchy, nodes, valid_positions)
    names = list(object_dict.keys())
    num_of_obj = len(names)
    names = ['h'] + names
    types = ['human'] + ['phyobj' for _ in range(num_of_obj)]
    state = pds.State([domain.types[t] for t in types], object_names=names)
    ctx = state.define_context(domain)
    predicates = generate_predicates(ctx, object_dict)

    predicates.append(ctx.is_working('h'))
    predicates.append(ctx.hand_empty('h'))
    predicates.append(ctx.human_at('h', 'floor'))

    ctx.define_predicates(predicates)
    translator = pds.strips.GStripsTranslator(domain, use_string_name=True)
    return state, translator, object_dict, object_names, names


def generation_process(data_dir, task_idx, goal_idx, quest_type, nodes, hierarchy, domain, valid_positions):
    """
    Generate one episode given goal template and quest type
    :param data_dir: path to save the raw json file
    :param task_idx: index of task under specific goal+quest_type setting
    :param goal_idx: index of goal template
    :param quest_type: type of instruction
    :return:
    """
    while True:
        print("Try task {}_{}_{}!".format(quest_type, goal_idx, task_idx))
        state, translator, object_dict, object_names, strips_names = \
            sample_and_define_scene(nodes, hierarchy, domain, valid_positions)
        cur_goal = generate_one_goal(object_dict, object_names, goal_idx)
        if not cur_goal.valid:
            print('Goal not valid!')
            continue
        if 'floor' not in cur_goal.rel_types['LOCATION']:
            cur_goal.rel_types['LOCATION'].append('floor')

        actions = generate_relevant_partially_grounded_actions(translator.domain, state, cur_goal.rel_types,
                                                               filter_static=True)
        strips_state = translator.compile_state(state)
        strips_operators = [translator.compile_operator(op, state, is_relaxed=False) for op in actions]
        extended_state = strips_state.copy()

        plan, extended_state, sub_plans = solve_goal(cur_goal, state, extended_state, strips_operators, translator)
        if plan is None:
            print('Goal not solvable!')
            continue
        if len(plan) == 0:
            print('Trivial task!')
            continue

        cur_extended_state = strips_state.copy()
        update_object_dict(object_dict, cur_extended_state, strips_names)
        new_data = Data(task_idx, object_dict, cur_goal.expr)


class GenerateTasksUnderGoal(object):
    def __init__(self, args, nodes, hierarchy, domain, valid_positions):
        self.dirname = args.dirname
        self.quest_type = args.quest_type
        self.nodes = nodes
        self.hierarchy = hierarchy
        self.domain = domain
        self.valid_positions = valid_positions

    def generate(self, pair):   # (task_idx, goal_idx)
        generation_process(self.dirname, pair[0], pair[1], self.quest_type,
                           self.nodes, self.hierarchy, self.domain, self.valid_positions)


if __name__ == '__main__':
    root = osp.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', type=str, default='raw_data', help='folder to save the raw json files')
    parser.add_argument('--quest_type', type=str, default='bring_me', help='type of instruction')
    args = parser.parse_args()

    nodes, hierarchy = hierarchy2json(filename=osp.join(root, 'sampling', 'object_hierarchy'), save_json=False)
    valid_positions = get_valid_positions(nodes, filename=osp.join(root, 'sampling', 'object_sample_space'))
    domain = pds.load_domain_file(root + 'domain.pddl')

    tasks = range(1000)
    goals = [0]
    pairs = list(itertools.product([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]))
    mp.Pool(8).map(
        GenerateTasksUnderGoal(args, nodes, hierarchy, domain, valid_positions).generate,
        pairs
    )

