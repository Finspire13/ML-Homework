# ML HW 3
import operator
from collections import defaultdict

CONF_OF_INTEREST = ['IJCAI', 'AAAI', 'COLT', 'CVPR',
                    'NIPS', 'KR', 'SIGIR', 'KDD']


def load_data():
    data_file = './FilteredDBLP.txt'
    with open(data_file) as file:
        raw = file.readlines()
    raw = [x.strip() for x in raw]

    data = []
    for i in range(len(raw)):
        if raw[i].startswith('#'):
            entry = {'author': [],
                     'title': None,
                     'year': None,
                     'conference': None}
        else:
            line = raw[i].split('\t')
            if line[0] == 'author':
                entry['author'].append(line[1])

            elif line[0] == 'Conference':
                # Filter out confs out of interest
                conf_index = [i in line[1].upper() for i in CONF_OF_INTEREST]
                if True in conf_index:
                    value = CONF_OF_INTEREST[conf_index.index(True)]
                    entry[line[0].lower()] = value
                    data.append(entry)
                else:
                    pass

            else:
                entry[line[0]] = line[1]

    return data


def get_filtered_data(data, key, values):
    if type(values) is not list:
        raise Exception('Values should be a list!')

    if key == 'author':
        return [i for i in data if not set(i[key]).isdisjoint(values)]
    else:
        return [i for i in data if i[key] in values]


def get_values(data, key):
    values = set()
    for i in range(len(data)):
        values.add(data[i][key])
    return values


def get_counts(data, key, values):
    return len(get_filtered_data(data, key, values))


def get_baskets(data, key):
    return [i[key] for i in data]


def get_sorted_patterns(patterns):
    return sorted(patterns.items(),
                  key=operator.itemgetter(1), reverse=True)


def get_team_data(data, team):
    return [i for i in data if set(i['author']) >= set(team)]


def remove_unclosed_items(patterns):
    keys_to_remove = []
    for key_a in patterns.keys():
        for key_b in patterns.keys():
            if set(key_a) < set(key_b):
                keys_to_remove.append(key_a)
    for key in keys_to_remove:
        patterns.pop(key)


def get_ranked_keyword_frequency(sentences, keywords):
    frequency_dist = defaultdict(lambda: 0)
    for keyword in keywords:
        for sentence in sentences:
            if keyword in sentence:
                frequency_dist[keyword] += 1
    return sorted(frequency_dist.items(),
                  key=operator.itemgetter(1), reverse=True)


def print_divider(char, txt):
    print('\n' + char * 30 + ' ' + txt + ' ' + char * 30 + '\n')
