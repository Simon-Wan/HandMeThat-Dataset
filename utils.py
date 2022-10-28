import itertools
from typing import Optional, Union, Callable, Tuple, Sequence, List, Mapping, Any, Dict
from pdsketch.interface.v2.value import ObjectType, NamedValueType, NamedValueTypeSlot
from pdsketch.interface.v2.state import State
from pdsketch.interface.v2.domain import Domain, OperatorApplier
# from hacl.pdsketch.interface.v2.planner.basic_planner import filter_static_grounding
from pdsketch.interface.v2.expr import ExpressionExecutionContext, is_simple_bool, get_simple_bool_def


def filter_static_grounding(domain, state, actions):
    output_actions = list()
    for action in actions:
        ctx = ExpressionExecutionContext(domain, state,
                                         bounded_variables=state.compose_bounded_variables(action.operator.arguments,
                                                                                           action.arguments))
        flag = True
        with ctx.as_default():
            for pre in action.operator.preconditions:
                if is_simple_bool(pre.bool_expr) and get_simple_bool_def(pre.bool_expr).static:
                    rv = pre.bool_expr.forward(ctx).item()
                    if rv < 0.5:
                        flag = False
                        break
        if flag:
            output_actions.append(action)
    return output_actions


def generate_predicates(ctx, object_dict):
    predicates = list()

    for name in object_dict.keys():
        # movable and receptacle
        if object_dict[name]['class'] != 'LOCATION':
            predicates.append(ctx.get_pred('movable')(name))
            if object_dict[name]['class'] == 'RECEPTACLE':
                predicates.append(ctx.get_pred('receptacle')(name))
        else:
            predicates.append(ctx.get_pred('receptacle')(name))
        # type
        predicates.append(ctx.get_pred('type-' + object_dict[name]['type'])(name))
        # ability
        for ability in object_dict[name]['ability']:
            if 'cleanable' in ability:
                tool = ability.split('-')[0]
                for tool_name in object_dict.keys():
                    if tool in tool_name:
                        predicates.append(ctx.valid_clean_pair(name, tool_name))
            else:
                predicates.append(ctx.get_pred(ability)(name))
        # location
        if 'inside' in object_dict[name].keys():
            predicates.append(ctx.inside(name, object_dict[name]['inside']))
        if 'ontop' in object_dict[name].keys():
            predicates.append(ctx.ontop(name, object_dict[name]['ontop']))
        # states
        for key in object_dict[name]['states'].keys():
            if key == 'color':
                predicates.append(ctx.get_pred('color-' + object_dict[name]['states'][key])(name))
            elif key == 'size':
                predicates.append(ctx.get_pred('size-' + object_dict[name]['states'][key])(name))
            else:
                if object_dict[name]['states'][key]:
                    predicates.append(ctx.get_pred(key)(name))

    return predicates


OTHER_ATTRIBUTES = ['cooked', 'dusty', 'frozen', 'stained', 'sliced', 'soaked', 'toggled', 'open']
OTHER_ABILITIES = ['cookable', 'dustyable', 'freezable', 'stainable', 'sliceable', 'soakable', 'toggleable', 'openable']
ACTION_ARGS = {
    'human-move': ['human', 'location', 'location'],
    'human-pick-up-at-location': ['human', 'movable', 'location'],
    'human-pick-up-from-receptacle-at-location': ['human', 'movable', 'receptacle', 'location'],
    'human-put-inside-location': ['human', 'movable', 'location'],
    'human-put-ontop-location': ['human', 'movable', 'location'],
    'human-put-inside-receptacle-at-location': ['human', 'movable', 'receptacle', 'location'],
    'human-put-ontop-receptacle-at-location': ['human', 'movable', 'receptacle', 'location'],
    'human-open-location': ['human', 'location'],
    'human-close-location': ['human', 'location'],
    'human-open-receptacle-at-location': ['human', 'receptacle', 'location'],
    'human-close-receptacle-at-location': ['human', 'receptacle', 'location'],
    'human-toggle-on-location': ['human', 'location'],
    'human-toggle-off-location': ['human', 'location'],
    'human-toggle-on-movable-at-location': ['human', 'tool', 'location'],
    'human-toggle-off-movable-at-location': ['human', 'tool', 'location'],
    'human-heat-obj': ['human', 'food', 'location'],
    'human-cool-obj': ['human', 'food', 'location'],
    'human-slice-obj': ['human', 'food', 'tool', 'location'],
    'human-soak-obj': ['human', 'tool', 'location'],
    'human-clean-obj-at-location': ['human', 'movable_cleanable', 'tool', 'location'],
    'human-clean-location': ['human', 'tool', 'location'],
    'robot-move-obj-to-human': ['human', 'movable', 'position'],
    'robot-move-obj-from-rec-into-rec': ['human', 'movable', 'position', 'position'],
    'robot-move-obj-from-rec-onto-rec': ['human', 'movable', 'position', 'position'],
    'robot-toggle-on': ['human', 'all'],
    'robot-toggle-off': ['human', 'all'],
    'robot-heat-obj': ['human', 'food'],
    'robot-cool-obj': ['human', 'food'],
    'robot-slice-obj': ['human', 'food'],
    'robot-soak-obj': ['human', 'other'],
    'robot-clean-obj': ['human', 'all'],
}


def generate_relevant_partially_grounded_actions(
        domain: Domain,
        state: State,
        rel_types: Dict,
        action_names: Optional[Sequence[str]] = None,
        action_filter: Optional[Callable[[OperatorApplier], bool]] = None,
        filter_static: Optional[bool] = True
) -> List[OperatorApplier]:
    if action_names is not None:
        action_ops = [domain.operators[x] for x in action_names]
    else:
        action_ops = domain.operators.values()
    rel = dict()
    rel['human'] = ['h']
    rel['location'] = rel_types['LOCATION']
    rel['receptacle'] = rel_types['RECEPTACLE']
    rel['food'] = rel_types['FOOD']
    rel['tool'] = rel_types['TOOL']
    rel['thing'] = rel_types['THING']
    rel['position'] = rel['location'] + rel['receptacle']
    rel['other'] = rel_types['FOOD'] + rel_types['TOOL'] + rel_types['THING']
    rel['movable'] = rel['receptacle'] + rel['other']
    rel['all'] = rel['movable'] + rel['location']
    rel['movable_cleanable'] = rel['receptacle'] + rel['thing']
    actions = list()
    for op in action_ops:
        argument_candidates = list()
        for idx, arg in enumerate(op.arguments):
            if isinstance(arg.type, ObjectType):
                candidates = state.object_type2name[arg.type.typename]
                type_class = ACTION_ARGS[op.name][idx]
                relevant_candidates = rel_analysis(candidates, rel[type_class])
                argument_candidates.append(relevant_candidates)
            else:
                assert isinstance(arg.type, NamedValueType)
                argument_candidates.append([NamedValueTypeSlot(arg.type)])
        for comb in itertools.product(*argument_candidates):
            actions.append(op(*comb))

    if filter_static:
        actions = filter_static_grounding(domain, state, actions)
    if action_filter is not None:
        actions = list(filter(action_filter, actions))
    return actions


def rel_analysis(candidates, relevant):
    result = list()
    for rel in relevant:
        if '#' not in rel:
            for cand in candidates:
                if cand[:len(rel)] == rel:
                    result.append(cand)
        else:
            result.append(rel)
    return result


def update_object_dict(object_dict, cur_extended_state, names):
    state = list(cur_extended_state)
    for predicate in state:
        if '_not' in predicate:
            continue
        if 'type' in predicate:
            continue
        if 'color' in predicate:
            continue
        if 'size' in predicate:
            continue
        args = predicate.split()
        if args[0] == 'inside':
            object_dict[names[int(args[1]) + 1]]['inside'] = names[int(args[2]) + 1]
            if 'ontop' in object_dict[names[int(args[1]) + 1]].keys():
                object_dict[names[int(args[1]) + 1]].pop('ontop')
            continue
        if args[0] == 'ontop':
            object_dict[names[int(args[1]) + 1]]['ontop'] = names[int(args[2]) + 1]
            if 'inside' in object_dict[names[int(args[1]) + 1]].keys():
                object_dict[names[int(args[1]) + 1]].pop('inside')
            continue
        for attr in OTHER_ATTRIBUTES:
            if args[0] == attr:
                object_dict[names[int(args[1]) + 1]]['states'][attr] = True
    return object_dict
