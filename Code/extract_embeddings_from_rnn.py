#!/usr/bin/env python
import sys, os
import torch
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../src/word_language_model')))
import data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract and plot LSTM weights')
parser.add_argument('-model', type=str, help='Meta file stored once finished training the corpus')
parser.add_argument('-v', '--vocabulary', default='../../../Data/LSTM/english_vocab.txt')
parser.add_argument('-o', '--output', default='Figures/verbs.png', help='Destination for the output figure')
parser.add_argument('-u', '--units', nargs='+', help='Which units in 2ND LAYER to plot')
parser.add_argument('-c', '--colors', nargs='+', help='corresponding colors for each unit in 2ND LAYER')
parser.add_argument('-f', '--units-first', nargs='+', help='Which units in 1st LAYER to plot')
parser.add_argument('-b', '--colors-first', nargs='+', help='corresponding colors for each unit in 1ST LAYER')
parser.add_argument('-i', '--input', default='../../../Data/Stimuli/singular_plural_verbs.txt',
					help='Text file with two tab delimited columns with the lists of output words to contrast with the PCA')
args = parser.parse_args()

if args.colors is not None:
	assert len(args.units) == len(args.colors), "!!!---Number of colors is not equal to number of units---!!!"
if args.colors_first is not None:
	assert len(args.units_first) == len(args.colors_first), "!!!---Number of colors_first is not equal to number of units_first (1st Layer)---!!!"

gate_names = ['Input', 'Forget', 'Cell', 'Output']
# Parse output dir and file names:
# os.makedirs(os.path.dirname(args.output), exist_ok=True)
dirname = os.path.dirname(args.output)
filename = os.path.basename(args.output)

# Load model
print('Loading models...')
print('\nmodel: ' + args.model+'\n')
model = torch.load(args.model, lambda storage, loc: storage)
model.rnn.flatten_parameters()
embeddings_in = model.encoder.weight.data.cpu().numpy()
embeddings_out = model.decoder.weight.data.cpu().numpy()
vocab = data.Dictionary(args.vocabulary)

# Read list of contrasted words (e.g., singular vs. plural verbs).
with open(args.input, 'r') as f:
	lines=f.readlines()
verbs_singular = [l.split('\t')[0].strip() for l in lines]
verbs_plural = [l.split('\t')[1].strip() for l in lines]
verbs_all = verbs_singular + verbs_plural
print('\nWords used (group 1):')
print(verbs_singular)
print('\nWords used (group 2):')
print(verbs_plural)

# Get index in the vocab for all words and extract embeddings
idx_verbs_singular = [vocab.word2idx[w] for w in verbs_singular]
idx_verbs_plural = [vocab.word2idx[w] for w in verbs_plural]
idx_verbs_all = idx_verbs_singular + idx_verbs_plural
embeddings_verbs_out_singular = embeddings_out[idx_verbs_singular, :]
embeddings_verbs_out_plural = embeddings_out[idx_verbs_plural, :]
embeddings_verbs_out_all = embeddings_out[idx_verbs_singular + idx_verbs_plural ,:]
y = '1'*len(idx_verbs_singular) + '2' * len(idx_verbs_plural)
y = np.asarray([int(j) for j in y])
# Also from encoder
embeddings_in_verbs_singular = embeddings_in[idx_verbs_singular, :]
embeddings_in_verbs_plural = embeddings_in[idx_verbs_plural, :]
embeddings_in_verbs_all = embeddings_in[idx_verbs_singular + idx_verbs_plural ,:]


#### Plot verb embeddings
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0, tol=1e-5, penalty='l2')
clf.fit(embeddings_verbs_out_all, y)
print('\nLeading units based on coefficents from a linearSVC')
print(650 + np.argsort(np.negative(np.abs(clf.coef_)))[0][::20])

pca = PCA(n_components=2)
pca.fit(embeddings_verbs_out_all)
pca_score = pca.explained_variance_ratio_
V = pca.components_
X_transformed = pca.fit_transform(embeddings_verbs_out_all)
PC1 = V[0, :]
PC2 = V[1, :]
print('\nLeading units based on PC1')
print(650 + np.argsort(np.negative(np.abs(PC1)))[0:20])

fig, ax = plt.subplots(1, figsize = (35, 20))
for i in tqdm(range(X_transformed.shape[0])):
	# if w % 10 == 1:
	ax.text(X_transformed[i, 0], X_transformed[i, 1], verbs_all[i], size=45)

lim_max = np.max(X_transformed)
lim_min = np.min(X_transformed)
ax.set_xlim((lim_min, lim_max))
ax.set_ylim((lim_min, lim_max))
ax.axis('off')
fig.savefig(os.path.join(dirname, 'PCA_'+filename))
print('Saved to: ' + os.path.join(dirname, 'PCA_'+filename))
plt.close(fig)

# Plot weights
units = [int(u) for u in args.units] # Second-layer units
bar_width = 0.2
## Extract weights from number units to verbs
fig, ax = plt.subplots(1, figsize = (15,10))
for u, from_unit in enumerate(units):
	# print(u, from_unit)
	if u == 0:
		label_sing = 'Singular form of verb'; label_plur = 'Plural form of verb'
	else:
		label_sing = ''; label_plur = ''
	from_unit = from_unit - 650
	output_weights_singular = embeddings_out[idx_verbs_singular, from_unit]
	ax.scatter(u + np.random.random(output_weights_singular.size) * bar_width - bar_width/2, output_weights_singular, s=400, color=args.colors[u], label=label_sing, marker='.')
	output_weights_plural = embeddings_out[idx_verbs_plural, from_unit]
	ax.scatter(u + np.random.random(output_weights_plural.size) * bar_width - bar_width/2, output_weights_plural, s=400, color=args.colors[u], label=label_plur, marker='_')
	print('Unit %i, SNR = %1.2f' % (from_unit+650, np.abs(np.mean(output_weights_singular)-np.mean(output_weights_plural))/(np.std(output_weights_singular)+np.std(output_weights_plural))))

units_first = []
if args.units_first:
	units_first = [int(u) for u in args.units_first] # First-layer units
	for u, to_unit in enumerate(units_first):
		print(u, to_unit)
		label_sing = ''; label_plur = ''
		to_unit = to_unit
		input_weights_singular = embeddings_in[idx_verbs_singular, to_unit]
		ax.scatter(len(units) + u + np.random.random(input_weights_singular.size) * bar_width - bar_width/2, input_weights_singular, s=400, color=args.colors_first[u], label=label_sing, marker='.')
		input_weights_plural = embeddings_in[idx_verbs_plural, to_unit]
		ax.scatter(len(units) + u + np.random.random(input_weights_plural.size) * bar_width - bar_width/2, input_weights_plural, s=400, color=args.colors_first[u], label=label_plur, marker='_')

plt.legend(fontsize=30, bbox_to_anchor=(1,1))
# plt.subplots_adjust(top=0.8)
plt.tick_params(axis='x', which='major', labelsize=35)
plt.tick_params(axis='y', which='major', labelsize=30)
plt.xticks(range(len(units)+len(units_first)), [str(u+1) for u in units] + [str(u+1) for u in units_first])
ax.set_ylabel('Weight size', fontsize = 35)
# ax.set_xlabel('Unit', fontsize = 35)
ax.axhline(linewidth=2, color='k', ls = '--')
fig.savefig(os.path.join(dirname, 'weight_dists_'+filename))
print('saved to: ' + os.path.join(dirname, 'weight_dists_'+filename))
plt.close(fig)

