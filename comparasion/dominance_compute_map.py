
#!/usr/bin/env python3
"""计算两个算法解集之间的支配关系并输出每个点对、每个近似算法的统计结果。

我的算法解集: route_planning/PSO_HP/dataset/DIMACS/result/merged_{LOC}_4obj.csv
近似算法解集: route_planning/approx_alg/dataset/result/{LOC}_{ALGO}.txt
输出文件: route_planning/comparison/dataset/dominance_{algo}.csv
"""
import csv
import os
import glob
from collections import defaultdict

OBJECTIVE_COLUMNS = ['distance', 'travel_time', 'elevation', 'avg_degree']
IGNORE_COLUMNS = {'solver', 'src', 'dest', 'source', 'destination', 'path', 'path_nodes'}


def read_solutions(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        objective_headers = [h for h in OBJECTIVE_COLUMNS if h in headers]
        if not objective_headers:
            objective_headers = [h for h in headers if h not in IGNORE_COLUMNS]
        for r in reader:
            src = r.get('src') or r.get('source')
            dest = r.get('dest') or r.get('destination')
            if src is None or dest is None:
                continue
            objs = []
            for h in objective_headers:
                if h in IGNORE_COLUMNS:
                    continue
                value = r.get(h, '')
                try:
                    objs.append(float(value))
                except Exception:
                    objs.append(float('nan'))
            rows.append({'src': src, 'dest': dest, 'objs': objs})
    return rows


def dominates(a, b):
    if not a or not b:
        return False
    n = min(len(a), len(b))
    strictly_better = False
    for i in range(n):
        if a[i] > b[i]:
            return False
        if a[i] < b[i]:
            strictly_better = True
    return strictly_better


def compute_pair_metrics(my_list, ref_list):
    my_dom_ref = my_be_dom_ref = my_non_inter = 0
    ref_dom_my = ref_be_dom_my = ref_non_inter = 0

    for my in my_list:
        dom_any = be_dom_any = False
        for ref in ref_list:
            if dominates(my, ref):
                dom_any = True
            if dominates(ref, my):
                be_dom_any = True
            if dom_any and be_dom_any:
                break
        if dom_any:
            my_dom_ref += 1
        if be_dom_any:
            my_be_dom_ref += 1
        if not dom_any and not be_dom_any:
            my_non_inter += 1

    for ref in ref_list:
        dom_any = be_dom_any = False
        for my in my_list:
            if dominates(ref, my):
                dom_any = True
            if dominates(my, ref):
                be_dom_any = True
            if dom_any and be_dom_any:
                break
        if dom_any:
            ref_dom_my += 1
        if be_dom_any:
            ref_be_dom_my += 1
        if not dom_any and not be_dom_any:
            ref_non_inter += 1

    return my_dom_ref, my_be_dom_ref, my_non_inter, ref_dom_my, ref_be_dom_my, ref_non_inter


def group_by_pair(rows):
    d = defaultdict(list)
    for r in rows:
        d[(r['src'], r['dest'])].append(r['objs'])
    return d


def find_ref_files(dataset_dir, loc):
    pattern = os.path.join(dataset_dir, 'result', f'{loc}_*.txt')
    return sorted(glob.glob(pattern))


def get_algo_name(path, loc):
    base = os.path.basename(path)
    prefix = f'{loc}_'
    if base.startswith(prefix):
        return os.path.splitext(base[len(prefix):])[0]
    return os.path.splitext(base)[0]


def write_pair_results(out_dir, loc, algo_name, rows):
    path = os.path.join(out_dir, f'dominance_{loc}_{algo_name}.csv')
    fieldnames = [
        'location', 'src', 'dest', 'my_size', 'ref_size',
        f'my_dom_{algo_name}', f'my_be_dom_{algo_name}', f'my_non_inter_{algo_name}',
        f'{algo_name}_dom_my', f'{algo_name}_be_dom_my', f'{algo_name}_non_inter'
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def main():
    base_my = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PSO_HP', 'dataset', 'DIMACS', 'result'))
    base_ref = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'approx_alg', 'dataset'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    os.makedirs(out_dir, exist_ok=True)

    locations = ['BAY', 'FLA', 'NY', 'COL']
    for loc in locations:
        my_path = os.path.join(base_my, f'merged_{loc}_4obj.csv')
        if not os.path.isfile(my_path):
            print('跳过我的算法文件:', my_path)
            continue
        my_group = group_by_pair(read_solutions(my_path))

        ref_files = find_ref_files(base_ref, loc)
        if not ref_files:
            print('未找到近似算法文件:', loc)
            continue

        for ref_path in ref_files:
            algo = get_algo_name(ref_path, loc)
            ref_group = group_by_pair(read_solutions(ref_path))
            common_pairs = sorted(set(my_group.keys()) & set(ref_group.keys()))
            if not common_pairs:
                print(f'[{loc}][{algo}] 无共同点对，跳过')
                continue

            rows = []
            for src, dest in common_pairs:
                my_list = my_group[(src, dest)]
                ref_list = ref_group[(src, dest)]
                my_dom_ref, my_be_dom_ref, my_non_inter, ref_dom_my, ref_be_dom_my, ref_non_inter = compute_pair_metrics(my_list, ref_list)
                rows.append({
                    'location': loc,
                    'src': src,
                    'dest': dest,
                    'my_size': len(my_list),
                    'ref_size': len(ref_list),
                    f'my_dom_{algo}': my_dom_ref,
                    f'my_be_dom_{algo}': my_be_dom_ref,
                    f'my_non_inter_{algo}': my_non_inter,
                    f'{algo}_dom_my': ref_dom_my,
                    f'{algo}_be_dom_my': ref_be_dom_my,
                    f'{algo}_non_inter': ref_non_inter,
                })

            output_path = write_pair_results(out_dir, loc, algo, rows)
            print(f'[{loc}][{algo}] 输出: {output_path}')


if __name__ == '__main__':
    main()
