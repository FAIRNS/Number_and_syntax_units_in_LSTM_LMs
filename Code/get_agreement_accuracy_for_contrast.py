import argparse, pickle
import numpy as np

parser = argparse.ArgumentParser(description='Get accuracy on agreement task for a specific contrast')
parser.add_argument('-ablation-results', required=True, help='Input ablation-results file')
parser.add_argument('-info', required=True, help='Input meta info file')
parser.add_argument('-condition', nargs='+', required=True, help='a list of key and their values, e.g., number_1=singular number_2=plural')
args = parser.parse_args()

condition_constraints = [c.split('=') for c in args.condition] # list of (key, value) tuples that defines the condition
print(condition_constraints)

ablation_results = pickle.load(open(args.ablation_results, 'rb'))
info = pickle.load(open(args.info, 'rb'))
success = []; p_difference = []
for i, sentence_info in enumerate(info):
    check_if_all_constraints_are_met = True
    for constraint in condition_constraints:
        key = constraint[0]; value = constraint[1]
        if sentence_info[key] != value: check_if_all_constraints_are_met = False
    if check_if_all_constraints_are_met:
        success.append(ablation_results['log_p_targets_correct'][i]>ablation_results['log_p_targets_wrong'][i])
        p_difference.append(np.exp(ablation_results['log_p_targets_correct'][i])-np.exp(ablation_results['log_p_targets_wrong'][i]))
if success:
    print(np.mean(success))
    print('p-difference: %1.5f +- %1.5f' % (np.mean(p_difference), np.std(p_difference)))
else:
    print('No sentences meet the keys-values in the condition. Check for typos and verify the meta info file include them')
