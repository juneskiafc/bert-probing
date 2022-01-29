from pathlib import Path
import pickle

root = Path('mlm_scores/MARC/marc_multi_mlm_head_only')
langs = ['de', 'es', 'fr']

for lang in langs:
    combined_scores = []
    for scores_file in root.rglob(f'{lang}*.pkl'):
        with open(scores_file, 'rb') as f:
            scores = pickle.load(f)
        print(scores_file, len(scores))
        combined_scores.extend(scores)

    combined_out_file = root.joinpath(f'{lang}_scores.pkl')
    with open(combined_out_file, 'wb') as fw:
        pickle.dump(combined_scores, fw)
    print(len(combined_scores))
    print('')

    