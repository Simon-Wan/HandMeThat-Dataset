import os
import os.path as osp
import json
import numpy as np

from text_env.text_utils import *
from text_env.robot_actions import *
from text_env.robot_space import *
from text_env.env import HMTEnv


def append_expert_demonstration(json_file):
    print(json_file, 'start!')
    demo_observations_partially = list()    # for Seq2Seq model
    demo_observations_fully = list()        # for Seq2Seq model
    demo_actions = list()
    env = HMTEnv(json_file, fully=False)
    fully_env = HMTEnv(json_file, fully=True)
    object_dict = env.game.current_object_dict.copy()
    target_idx = np.random.choice(range(len(env.game.get_objects_in_meaning())))
    target = env.game.get_objects_in_meaning()[target_idx]
    target_object = target[0]
    if env.game.utterance[0] == 'bring_me':
        demo_actions += find_and_pick_up_obj(env, object_dict, target_object)
        demo_actions += [
            'move to human',
            'give {} to human'.format(env.wrap(target_object, label=True))
        ]
    elif env.game.utterance[0] == 'move_to':
        target_position = target[1]
        demo_actions += find_and_pick_up_obj(env, object_dict, target_object)
        demo_actions += move_to_pos(env, object_dict, target_object, target_position)
    elif env.game.utterance[0] == 'change_state':
        verb = target[1]
        if verb == 'soak':
            demo_actions += find_and_pick_up_obj(env, object_dict, target_object)
            demo_actions.append('move to sink')
            demo_actions.append('put {} into sink'.format(env.wrap(target_object, label=True)))
            if not object_dict['sink']['states']['toggled']:
                demo_actions.append('toggle on sink')
            demo_actions.append('soak {}'.format(env.wrap(target_object, label=True)))
        elif verb == 'heat':
            demo_actions += find_and_pick_up_obj(env, object_dict, target_object)
            pos = 'stove'
            demo_actions.append('move to {}'.format(pos))
            demo_actions.append('put {} onto {}'.format(env.wrap(target_object, label=True), pos))
            if not object_dict[pos]['states']['toggled']:
                demo_actions.append('toggle on {}'.format(pos))
            demo_actions.append('heat {}'.format(env.wrap(target_object, label=True)))
        elif verb == 'cool':
            demo_actions += find_and_pick_up_obj(env, object_dict, target_object)
            pos = 'refrigerator'
            demo_actions.append('move to {}'.format(pos))
            if not object_dict[pos]['states']['open']:
                demo_actions.append('open {}'.format(pos))
            demo_actions.append('put {} into {}'.format(env.wrap(target_object, label=True), pos))
            demo_actions.append('close {}'.format(pos))
            if not object_dict[pos]['states']['toggled']:
                demo_actions.append('toggle on {}'.format(pos))
            demo_actions.append('cool {}'.format(env.wrap(target_object, label=True)))
        elif verb == 'open':
            demo_actions += find_obj(env, object_dict, target_object)
            demo_actions.append('open {}'.format(env.wrap(target_object, label=True)))
        elif verb == 'toggle':
            demo_actions += find_obj(env, object_dict, target_object)
            demo_actions.append('toggle on {}'.format(env.wrap(target_object, label=True)))
        elif verb == 'clean':
            if object_dict[target_object]['class'] != 'LOCATION':
                demo_actions += find_and_pick_up_obj(env, object_dict, target_object)
                demo_actions.append('move to table')
                demo_actions.append('put {} onto table'.format(env.wrap(target_object, label=True)))
            if object_dict[target_object]['subclass'] in ['vessel', 'tableware', 'utensil', 'tray']:
                tool = 'dishtowel#0'
            else:
                tool = 'rag#0'
            demo_actions += find_and_pick_up_obj(env, object_dict, tool)
            demo_actions.append('move to sink')
            demo_actions.append('put {} into sink'.format(env.wrap(tool, label=True)))
            if not object_dict['sink']['states']['toggled']:
                demo_actions.append('toggle on sink')
            demo_actions.append('soak {}'.format(env.wrap(tool, label=True)))
            demo_actions.append('pick up {}'.format(env.wrap(tool, label=True)))
            if object_dict[target_object]['class'] != 'LOCATION':
                demo_actions.append('move to table')
                demo_actions.append('clean {} with {}'.format(env.wrap(target_object, label=True), env.wrap(tool, label=True)))
            else:
                demo_actions.append('move to {}'.format(target_object))
                demo_actions.append('clean {} with {}'.format(target_object, env.wrap(tool, label=True)))
        elif verb == 'slice':
            demo_actions += find_and_pick_up_obj(env, object_dict, target_object)
            demo_actions.append('move to table')
            demo_actions.append('put {} onto table'.format(env.wrap(target_object, label=True)))
            tool = 'knife#0'
            demo_actions += find_and_pick_up_obj(env, object_dict, tool)
            demo_actions.append('move to table')
            demo_actions.append('slice {} with {}'.format(env.wrap(target_object, label=True), env.wrap(tool, label=True)))
        else:
            raise ValueError
    else:
        raise ValueError
    # no need to remove '#' in HMTEnv
    demo_actions = [a.replace('#', ' ') for a in demo_actions]  # simplify: remove '#'

    obs, info = fully_env.reset()
    idx_start = obs.find('The human agent has')
    idx_end = obs.find('Now you are')
    obs = obs[:idx_start] + obs[idx_end:]       # remove task_description in initial_obs
    obs = obs.replace('#', ' ')
    obs = obs.replace(',', '')
    demo_observations_fully.append(obs)
    for a in demo_actions:
        ob, _, done, _ = fully_env.step(a)
        ob = ob.replace('#', ' ')        # simplify: remove '#'
        ob = ob.replace(',', '')
        if not done:
            obs += ' [SEP] ' + a + ' [SEP] '
            obs += ob
            demo_observations_fully.append(obs)
        else:
            break
    if len(demo_actions) != len(demo_observations_fully):
        raise ValueError

    obs, info = env.reset()
    idx_start = obs.find('The human agent has')
    idx_end = obs.find('Now you are')
    obs = obs[:idx_start] + obs[idx_end:]  # remove task_description in initial_obs
    obs = obs.replace('#', ' ')
    obs = obs.replace(',', '')
    demo_observations_partially.append(obs)
    for a in demo_actions:
        ob, _, done, _ = env.step(a)
        ob = ob.replace('#', ' ')  # simplify: remove '#'
        ob = ob.replace(',', '')
        if not done:
            obs += ' [SEP] ' + a + ' [SEP] '
            obs += ob + env.get_look(fully=False).replace('#', ' ')
            demo_observations_partially.append(obs)
        else:
            break
    if len(demo_actions) != len(demo_observations_partially):
        raise ValueError

    task_desc = env.get_task_description()
    task_desc = task_desc.replace('#', ' ')
    data = load_from_json(json_file)
    data.demo_actions = demo_actions
    data.demo_observations_fully = demo_observations_fully
    data.demo_observations_partially = demo_observations_partially
    data.task_description = task_desc

    return data


def find_obj(env, object_dict, target_object):
    actions = list()
    if 'inside' in object_dict[target_object].keys():
        pos = object_dict[target_object]['inside']
    elif 'ontop' in object_dict[target_object].keys():
        pos = object_dict[target_object]['ontop']
    else:
        raise ValueError
    if object_dict[pos]['class'] == 'LOCATION':
        actions.append('move to {}'.format(pos))
        if 'openable' in object_dict[pos]['ability'] and not object_dict[pos]['states']['open']:
            actions.append('open {}'.format(pos))
    else:
        if 'inside' in object_dict[pos].keys():
            loc = object_dict[pos]['inside']
        elif 'ontop' in object_dict[pos].keys():
            loc = object_dict[pos]['ontop']
        else:
            raise ValueError
        if object_dict[loc]['class'] == 'LOCATION':
            actions.append('move to {}'.format(loc))
            if 'openable' in object_dict[loc]['ability'] and not object_dict[loc]['states']['open']:
                actions.append('open {}'.format(loc))
            if 'openable' in object_dict[pos]['ability'] and not object_dict[pos]['states']['open']:
                actions.append('open {}'.format(pos))
        else:
            raise ValueError
    return actions


def find_and_pick_up_obj(env, object_dict, target_object):
    actions = list()
    if 'inside' in object_dict[target_object].keys():
        pos = object_dict[target_object]['inside']
    elif 'ontop' in object_dict[target_object].keys():
        pos = object_dict[target_object]['ontop']
    else:
        raise ValueError
    if object_dict[pos]['class'] == 'LOCATION':
        actions.append('move to {}'.format(pos))
        if 'openable' in object_dict[pos]['ability'] and not object_dict[pos]['states']['open']:
            actions.append('open {}'.format(pos))
        actions.append('pick up {}'.format(env.wrap(target_object, label=True)))
    else:
        if 'inside' in object_dict[pos].keys():
            loc = object_dict[pos]['inside']
        elif 'ontop' in object_dict[pos].keys():
            loc = object_dict[pos]['ontop']
        else:
            raise ValueError
        if object_dict[loc]['class'] == 'LOCATION':
            actions.append('move to {}'.format(loc))
            if 'openable' in object_dict[loc]['ability'] and not object_dict[loc]['states']['open']:
                actions.append('open {}'.format(loc))
            if 'openable' in object_dict[pos]['ability'] and not object_dict[pos]['states']['open']:
                actions.append('open {}'.format(pos))
            actions.append(
                'pick up {} from {}'.format(env.wrap(target_object, label=True), env.wrap(pos, label=True)))
        else:
            raise ValueError
    return actions


def move_to_pos(env, object_dict, target_object, target_position):
    actions = list()
    if 'has-inside' in object_dict[target_position]['ability']:
        prep = 'into'
    elif 'has-ontop' in object_dict[target_position]['ability']:
        prep = 'onto'
    else:
        raise ValueError
    if object_dict[target_position]['class'] == 'LOCATION':
        actions.append('move to {}'.format(target_position))
        if 'openable' in object_dict[target_position]['ability'] and not object_dict[target_position]['states']['open']:
            actions.append('open {}'.format(target_position))
        actions.append('put {} {} {}'.format(env.wrap(target_object, label=True), prep, target_position))
    else:
        if 'inside' in object_dict[target_position].keys():
            loc = object_dict[target_position]['inside']
        elif 'ontop' in object_dict[target_position].keys():
            loc = object_dict[target_position]['ontop']
        else:
            raise ValueError
        actions.append('move to {}'.format(loc))
        if 'openable' in object_dict[loc]['ability'] and not object_dict[loc]['states']['open']:
            actions.append('open {}'.format(loc))
        if 'openable' in object_dict[target_position]['ability'] and not object_dict[target_position]['states']['open']:
            actions.append('open {}'.format(target_position))
        actions.append('put {} {} {}'.format(env.wrap(target_object, label=True), prep, env.wrap(target_position, label=True)))

    return actions


def generate_expert_demonstrations(raw_data_dir='./raw_data', output_dir='./expert_data'):
    if not osp.exists(raw_data_dir):
        raise ValueError('Raw data not exists!')
    if not osp.exists(output_dir):
        print('Create new folder for expert demonstrations!')
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            file_name = file.split('_')
            new_file_name = '_'.join([file_name[1], file_name[0], file_name[2], file_name[3], file_name[4]])
            try:
                with open(osp.join(output_dir, new_file_name), 'r') as f:
                    print(new_file_name, 'already exists!')
                    continue
            except:
                pass

            try:
                data = append_expert_demonstration(osp.join(raw_data_dir, file_name[1], file))
            except ValueError as e:
                print(e, file_name)
                continue
            new_file_name = '_'.join([file_name[1], file_name[0], file_name[2], file_name[3], file_name[4]])
            with open(osp.join(output_dir, new_file_name), 'w+') as f2:
                json.dump(data.__dict__, f2)


if __name__ == '__main__':
    generate_expert_demonstrations()

