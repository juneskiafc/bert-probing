import yaml

for dataset in [
    'ar',
    'bg',
    'de',
    'el',
    'en',
    'es',
    'fr',
    'hi',
    'ru',
    'sw',
    'th',
    'tr',
    'ur',
    'vi',
    'zh'
]:
    yaml_dict = {
        dataset: {
            'data_format': "PremiseAndOneHypothesis",
            "enable_san": False,
            "split_names": ['test'],
            "labels": ['contradiction', 'neutral', 'entailment'],
            "metric_meta": ['ACC'],
            'loss': 'CeCriterion',
            'n_class': 3,
            'task_type': 'Classification'
        }
    }
    out_file = f'experiments/NLI/{dataset}/task_def.yaml'
    with open(out_file, 'w') as f:
        dumped = yaml.dump(yaml_dict, f)
