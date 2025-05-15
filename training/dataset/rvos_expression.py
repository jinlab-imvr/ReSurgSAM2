def get_expression_and_category(category, action_id, location_id=0, dataset_name='Ref-Endovis18', is_unique=True):
    linking_verb = 'are' if category[-1] == 's' else 'is'
    if '17' in dataset_name:
        if is_unique:
            exp = '{} {} {}'.format(category, linking_verb, action_mapping[action_id])
        else:
            exp = '{} {} {} on the {}'.format(category, linking_verb, action_mapping[action_id], location_mapping[location_id])
    elif '18' in dataset_name:
        if action_id == 0:   # for tissue
            exp = '{}'.format(category)
        else:   # for instrument
            exp = '{} {} {}'.format(category, linking_verb, action_mapping[action_id])
    else:
        raise ValueError('Unknown dataset name')
    category_id = category_mapping[category]
    return exp, category_id

category_mapping = {
    'bipolar forceps': 1, 'prograsp forceps': 2,'large needle driver': 3, 'monopolar curved scissors': 4,
    'suction': 5, 'clip applier': 6, 'ultrasound probe': 7, 'vessel sealer': 8, 'grasping retractor': 9,
    'kidney-parenchyma': 10, 'covered-kidney': 11, 'small-intestine': 12
}


action_mapping = {
    1: 'idle', 2: 'manipulating tool', 3: 'performing knot tying', 4: 'suturing tissue', 5: 'manipulating tissue',
    6: 'grasping tissue', 7: 'retracting tissue', 8: 'cutting tissue', 9: 'suctioning blood', 10: 'clipping tissue',
    11: 'sensing vessels', 12: 'transecting vessel', 13: 'cauterizing tissue'
}

location_mapping = {
    1: 'bottom left', 2: 'bottom center', 3: 'bottom right',
    4: 'left', 5: 'center', 6: 'right',
    7: 'top left', 8: 'top center', 9: 'top right'
}