from pathlib import Path
import jsonlines
import pandas as pd

tasks = ['BERT', 'MARC', 'NER', 'POS', 'NLI', 'PAWSX']
for task in tasks:
    for ds_task in tasks:
        if ds_task != 'BERT':
            for csv_file in Path('head_probe_outputs').joinpath(task, ds_task).rglob("*.csv"):
                if len(csv_file.name.split('-')) == 1:
                    data = pd.read_csv(csv_file, index_col=0)
                    out_file = csv_file.parent.joinpath(csv_file.with_suffix('.jsonl').name)
                    with jsonlines.open(out_file, 'w') as writer:
                        with open(csv_file, 'r') as fr:
                            for i, row in enumerate(fr):
                                if i > 0:
                                    split_row = row.split(',')
                                    val = split_row[1:]
                                    if len(split_row) > 2:
                                        val = [int(v.strip(' ').replace('"', "").replace("[", "").replace("]", "").strip("\n")) for v in val]
                                    else:
                                        val = val[0]
                                        val = int(val.strip('\n').strip(' '))
                                    writer.write(val)
                # csv_file.unlink()
            print(task, ds_task)
