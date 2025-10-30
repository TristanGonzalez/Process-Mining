from decision_mining_ml import DecisionMiningML
import json


def main():
    csv_path = "data/road_traffic_fine_.csv"
    dm = DecisionMiningML(csv_path)
    # aggressive pruning parameters for readability
    res = dm.train_for_all(test_size=0.3, prune_depth=3, prune_min_samples_leaf=50)

    out = {}
    for k, v in res.items():
        if isinstance(v, dict) and 'error' in v:
            out[k] = {'error': v['error']}
        else:
            out[k] = {
                'n_samples': v.get('n_samples'),
                'eval_default': v.get('eval_default'),
                'eval_pruned': v.get('eval_pruned'),
                'viz_default': v.get('viz_default'),
                'rules_default': v.get('rules_default')
            }

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
