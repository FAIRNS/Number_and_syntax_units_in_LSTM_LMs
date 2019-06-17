import pickle, argparse
import numpy as np

parser = argparse.ArgumentParser(description='Add additional keys (success and perplexity) to meta info')
parser.add_argument('-i', '--info', required=True, help='Input meta info pkl file')
parser.add_argument('-r', '--ablation_results', required=True, help='pkl file which is the output of the ablation-experiment script')
parser.add_argument('-a', '--activations', required=True, help='pkl file with the activations')
args = parser.parse_args()

# Load info
with open(args.info, 'rb') as f:
    info = pickle.load(f)

# Load ablation results
with open(args.ablation_results, 'rb') as f:
    ablation_results = pickle.load(f)
correct_wrong = ablation_results['log_p_targets_correct'] > ablation_results['log_p_targets_wrong']

# Load activations file and calculate perplexity for each sentence
activations = pickle.load(open(args.activations, 'rb'))
log_probs = activations['log_probabilities']
perplexities = [np.exp(-sent_log_probs.mean()) for sent_log_probs in log_probs]

# Add to meta info file
info_with_success = []
for curr_info, success, perp in zip(info, correct_wrong, perplexities):
    if success:
        curr_info['success'] = 'correct'
    else:
        curr_info['success'] = 'wrong'

    curr_info['perplexity'] = perp

    info_with_success.append(curr_info)

# Save new info
with open(args.info, 'wb') as f:
    pickle.dump(info_with_success, f)
print('The info file was OVERWRITTEN by the new one that contains success and perplexity: ' + args.info)
