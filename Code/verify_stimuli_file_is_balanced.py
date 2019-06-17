import argparse

# Output is a tab-delimited list of stimuli with info: sentence \t tense \t subject gender \t subject number

# Parse arguments
parser = argparse.ArgumentParser(description='Stimulus generator for Italian')
parser.add_argument('-f', '--data-filename', default='objrel_4000.txt', type=str, help = 'filename of the dataset to be examined')
args = parser.parse_args()


with open(args.data_filename, 'r') as f:
    stimuli = f.readlines()

features = [s.split('\t')[2::] for s in stimuli]
for i in range(len(features[0])):
    print('\nFeature #%i' % (i+1))
    curr_feature_values = [fs[i] for fs in features]
    value_strs = list(set(curr_feature_values))
    for v, val in enumerate(value_strs):
        num_curr_val = sum([1 if curr_val == val else 0 for curr_val in curr_feature_values])
        print(val.strip(), num_curr_val)