import jacinle
import os.path as osp
import pdsketch as pds

import argparse
import multiprocessing as mp
import itertools
import numpy as np

from utils import *
from sampling.scene_sampler import hierarchy2json, get_valid_positions, sample_scene
from sampling.goal_sampler import generate_one_goal
from planning.trajectory_generator import solve_goal
from data_processing.data_class import Data, generate_json
from pragmatic_reasoning.meaning import generate_reasonable_meaning, prob_m_given_g, get_robot_operators_upon_goal, reformat_meanings
from pragmatic_reasoning.utterance import get_all_utterances, remove_low_probs
from pragmatic_reasoning.rsa import sample_meaning, get_utterance, estimate_meaning


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

        # the subgoal list is shuffled in solve_goal
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

        # random truncate
        remaining_steps = len(plan)
        if remaining_steps < 5:
            stop_step = 0
        else:
            stop_step = np.random.randint((remaining_steps - 4) / 2, remaining_steps - 4)
        solution_idx = 0
        for subgoal_idx, sub_plan in enumerate(sub_plans):
            for simplified_op in sub_plan:
                if solution_idx >= stop_step and 'hand-empty 0' in cur_extended_state:  # todo: the rule of break
                    break
                if solution_idx >= len(plan):
                    break
                extended_op = None
                for op in strips_operators:
                    if simplified_op.raw_operator.name == op.raw_operator.name and simplified_op.raw_operator.arguments == op.raw_operator.arguments:
                        extended_op = op
                        break
                cur_extended_state = extended_op.apply(cur_extended_state)
                new_data.append_action(extended_op.raw_operator.name, list(extended_op.raw_operator.arguments))
                new_data.set_subgoal(cur_goal.expr_list[subgoal_idx])
                remaining_steps -= 1
                solution_idx += 1
        # start query
        if remaining_steps <= 1:
            print('Remaining steps not enough!')
            continue

        # compile the goal after human actions
        strips_goal = translator.compile_expr(cur_goal.expr, state)
        strips_goal = strips_goal[0]
        cur_task = pds.strips.GStripsTask(cur_extended_state, strips_goal, strips_operators, is_relaxed=False)
        cur_task = translator.relevance_analysis(cur_task)
        cur_task = cur_task.compile()
        heuristic = pds.strips.StripsHFFHeuristic(cur_task, translator)
        hff = heuristic.compute(cur_task.state)

        # state for robot
        new_state = []
        for predicate in cur_extended_state:
            if predicate != 'is-working 0' and predicate != 'is-waiting_not 0':
                new_state.append(predicate)
        new_state.append('is-waiting 0')
        new_state.append('is-working_not 0')
        cur_extended_state = frozenset(new_state)
        object_dict = update_object_dict(object_dict, cur_extended_state, strips_names)
        new_data.current_object_dict = object_dict.copy()

        # get meanings
        meaning_list = generate_reasonable_meaning(object_dict, cur_goal.rel_targets['POSITION'], quest_type)
        robot_operators_dict, reward_dict, useful_objects = get_robot_operators_upon_goal(
            cur_extended_state, translator, strips_operators, strips_goal, remaining_steps, cur_goal.rel_targets, hff,
            quest_type)
        # useful_objects is A(g)

        # compute P(m|g)
        prob_list = prob_m_given_g(meaning_list, reward_dict, object_dict, useful_objects, remaining_steps,
                                   cur_goal.rel_targets['POSITION'])
        # list all meanings and utterances
        meanings, probs = reformat_meanings(meaning_list, prob_list)
        utterances = get_all_utterances(meanings)

        if sum(probs) == 0:
            print('P(m|g) compute error!')
            continue

        short_meanings, short_probs, short_utterances = remove_low_probs(meanings, probs, utterances, hierarchy)
        if not short_meanings:
            print('Empty meaning pool error!')
            continue

        meaning_idx, meaning = sample_meaning(short_meanings, short_probs)
        # meaning_idx is the index in short_meanings

        objects_in_meaning = get_objects_from_description(object_dict, meaning)
        # objects_in_meaning is A(m)

        utterance_idx, utterance = get_utterance(meaning_idx, short_meanings, short_utterances, hierarchy, short_probs)

        objects_in_utterance = get_objects_from_description(object_dict, utterance,
                                                            cur_goal.rel_targets['POSITION'])  # todo: !!!
        # objects_in_meaning is A(u)

        rsa_meaning = estimate_meaning(utterance_idx, short_utterances, short_meanings, hierarchy, short_probs)

        objects_in_rsa_meaning = get_objects_from_description(object_dict, rsa_meaning)
        # objects_in_rsa_meaning is A(r)

        level = get_hardness_level(useful_objects[quest_type], objects_in_meaning, objects_in_utterance,
                                   objects_in_rsa_meaning)
        new_data.level = level

        new_data.set_utterance(utterance)
        possible_solution = []
        for op in plan:
            possible_solution.append({'name': op.raw_operator.name, 'arguments': list(op.raw_operator.arguments)})
        new_data.set_private(meaning, possible_solution, rsa_meaning)

        new_data.set_answer_objects(objects_in_utterance, objects_in_meaning, useful_objects[quest_type],
                                    objects_in_rsa_meaning)
        new_data.goal_idx = goal_idx
        generate_json(new_data, quest_type, root)
        print("Task {}_{}_{} finished!".format(quest_type, goal_idx, task_idx))
        task_idx += 1
        return new_data


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

