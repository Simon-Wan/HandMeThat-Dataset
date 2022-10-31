import json
import os
import os.path as osp


def create_split():
    type_split = {
        'bring_me': list(),
        'move_to': list(),
        'change_state': list(),
    }
    level_split = {
        'level1': list(),
        'level2': list(),
        'level3': list(),
        'level4': list(),
    }
    goal_split = dict()
    for idx in range(69):
        goal_split[str(idx)] = list()
    return type_split, level_split, goal_split


def add_file(type_split, level_split, goal_split, filename):
    quest_type, goal_idx, task_idx, hardness_level = filename[5:-5].split('-')
    type_split[quest_type].append(filename)
    goal_split[goal_idx].append(filename)
    if hardness_level == '1':
        level_split['level1'].append(filename)
    elif hardness_level == '2':
        level_split['level2'].append(filename)
    elif hardness_level == '3':
        level_split['level3'].append(filename)
    elif hardness_level == '0':
        level_split['level4'].append(filename)


def print_count(type_split, level_split, goal_split):
    print('bring_me: {}, move_to: {}, change_state: {}'.format(
        len(type_split['bring_me']), len(type_split['move_to']), len(type_split['change_state'])
    ))
    print('level1: {}, level2: {}, level3: {}, level4: {}'.format(
        len(level_split['level1']), len(level_split['level2']), len(level_split['level3']), len(level_split['level4'])
    ))
    print('goal:')
    print([len(goal_split[str(idx)]) for idx in range(69)])


if __name__ == '__main__':
    type_split, level_split, goal_split = create_split()
    data_info = './data_info.json'
    data_dir = './expert_data'
    if not osp.exists(data_dir):
        raise ValueError('Raw data not exists!')

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # task - quest type - goal idx - task idx - hardness level
            add_file(type_split, level_split, goal_split, file)

    print_count(type_split, level_split, goal_split)
    with open(data_info, 'w+') as f:
        json.dump([type_split, level_split, goal_split], f)
            

