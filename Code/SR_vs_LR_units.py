import os, mne, argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import GeneralizingEstimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generalization across time')
parser.add_argument('-s', '--sentences', type=str, help='Path to text file containing the list of sentences to analyze')
parser.add_argument('-m', '--metadata', type=str, help='The corresponding meta data of the sentences')
parser.add_argument('-a', '--activations', '--LSTM-file-name', type=str, help='The corresponding sentence (LSTM) activations')
parser.add_argument('-g', '--gate', type=str, help='One of: gates.in, gates.forget, gates.out, gates.c_tilde, hidden, cell')
parser.add_argument('-o', '--output-file-name', type=str, help='Path to output folder for figures')
args = parser.parse_args()


def generate_events_object(metadata):
    events = np.empty((0, 3), dtype = int)
    cnt = 0; IX = []
    for i, curr_info in enumerate(metadata):
        if curr_info['number_1'] == 'plural' and curr_info['number_2'] == 'singular': # PS: event = 1
            curr_line = np.asarray([cnt, 0, 2])
            events = np.vstack((events, curr_line)); cnt += 1
            IX.append(i)
        elif curr_info['number_1'] == 'singular' and curr_info['number_2'] == 'plural': # SP: event = 2
            curr_line = np.asarray([cnt, 0, 1])
            events = np.vstack((events, curr_line)); cnt += 1
            IX.append(i)

    event_id = dict(SP = 1, PS=2)
    return events, event_id, IX


def generate_epochs_object(data, sampling_rate = 1):
    n_channels = data[0].shape[0]
    info = mne.create_info(n_channels, sampling_rate)
    events, event_id, IX = generate_events_object(metadata)
    epochs = mne.EpochsArray(data[IX, :, :], info, events, 0, event_id)
    return epochs, events, IX


def get_scores_from_gat(epochs, seed):
    from sklearn.svm import LinearSVC
    X_train, X_test, y_train, y_test = train_test_split(epochs.get_data(), epochs.events[:, 2] == 2, test_size=0.2,
                                                        random_state=seed)
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, penalty='l2'))
    # clf =
    time_gen = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=-2)
    time_gen.fit(X_train, y_train)
    scores = time_gen.score(X_test, y_test)

    print('Units with highest weights of a classifier trained to predict subject''s number:')
    print([(i, j) for (i, j) in zip(np.transpose(np.argsort(np.negative(np.abs(time_gen.estimators_[1]._final_estimator.coef_))))[0:20],
                                    np.transpose(np.sort(np.negative(np.abs(time_gen.estimators_[1]._final_estimator.coef_))))[0:20])])


    return time_gen, scores

def get_gat_scores(data, omit_units, seed=42):
    # Generalization across time
    data_reduced = np.delete(data, omit_units, 1)
    epochs, _, _ = generate_epochs_object(data_reduced)
    time_gen, scores_reduced_model = get_scores_from_gat(epochs, seed)

    return scores_reduced_model

print(args)
#### Load data
metadata = pickle.load(open(args.metadata, 'rb'))
data = pickle.load(open(args.activations, 'rb'))
data = np.dstack(data[args.gate])
data = np.moveaxis(data, 2, 0) # num_trials X num_units X num_timepoints
with open(args.sentences, 'r') as f:
    sentences = f.readlines()
sentences = [s.split(' ') for s in sentences]

#### Plot Full model
fig1, ax1 = plt.subplots(1, figsize=(18,10))
# omit_units = [] # Omit nothing (Full model)
# title = 'Full model'
# print('model for: ' + title)
# scores_full_model = get_gat_scores(data, omit_units)
# scores_full_model = scores_full_model[1, :] # Scores only when trained on subject (first noun)
# ax1.errorbar(x=range(scores_full_model.shape[0]), y=scores_full_model, linewidth=1.5, label=title, color='k')

#### Plot Full model - LR units
omit_units = [775, 987] # Omit nothing (Full model)
title = 'Full-model minus LR-units'
print('model for: ' + title)
scores_full_model = [] # list of scores from all random CV seeds
for seed in range(5):
    scores = get_gat_scores(data, omit_units, seed)
    scores_full_model.append(scores[1, :]) # Scores only when trained on subject (first noun)
scores_full_model = np.vstack(scores_full_model)
ax1.errorbar(x=range(scores_full_model.shape[1]), y=np.mean(scores_full_model, axis=0), yerr=np.std(scores_full_model, axis=0), linewidth=3, label=title, color='k', ls='-')

#### LR units
colors = ['b', 'r']
line_styles = [':', '--']
scores_LR_units = []
for u, unit in enumerate([775, 987]):
    print('unit ' + str(unit))
    title = 'Unit ' + str(unit+1) +' (LR)'
    omit_units = list(set(range(data.shape[1])) - set([unit]))
    scores_LR_unit = []
    for seed in range(5):
        scores = get_gat_scores(data, omit_units, seed)
        scores_LR_unit.append(scores[1, :]) # Scores only when trained on subject (first noun)
    scores_LR_unit = np.vstack(scores_LR_unit)
    jitter = 0
    if u == 1: jitter = 0.01
    ax1.errorbar(x=range(scores_LR_unit.shape[1]), y=np.mean(scores_LR_unit, axis=0)+jitter, yerr=np.std(scores_LR_unit, axis=0), linewidth=6, label=title, color=colors[u], ls=line_styles[u])
    scores_LR_units.append(scores_LR_unit)

#### Plot 2nd layer
# omit_units = list(range(650)) # Omit nothing (Full model)
# title = 'Second layer'
# print('model for: ' + title)
# scores_2nd_layer = get_gat_scores(data, omit_units)
# scores_2nd_layer = scores_2nd_layer[1, :] # Scores only when trained on subject (first noun)
# ax1.errorbar(x=range(scores_2nd_layer.shape[0]), y=scores_2nd_layer, linewidth=0.5, label=title, color='c')

#### Plot 2nd layer - LR units
# omit_units = list(range(650)) + [775, 987] # Omit nothing (Full model)
# title = 'Second layer - LR unit'
# print('model for: ' + title)
# scores_2nd_layer = get_gat_scores(data, omit_units)
# scores_2nd_layer = scores_2nd_layer[1, :] # Scores only when trained on subject (first noun)
# ax1.errorbar(x=range(scores_2nd_layer.shape[0]), y=scores_2nd_layer, linewidth=0.5, label=title, color='c', ls='--')

#### Single units
# scores_all_single_units = []
# LG_units = []
# for unit in tqdm(range(1300)):
#     print('unit ' + str(unit))
#     omit_units = list(set(range(data.shape[1])) - set([unit]))
#     scores_single_unit = get_gat_scores(data, omit_units)
#     scores_single_unit = scores_single_unit[1, :] # Scores only when trained on subject (first noun)
#     scores_from_subject_to_2nd_noun = scores_single_unit[1:5]
#     if all(scores_from_subject_to_2nd_noun>0.99):
#         LG_units.append((unit, scores_single_unit))
#     scores_all_single_units.append(scores_single_unit)
# print(LG_units)
# scores_all_single_units = np.vstack(scores_all_single_units)
# ax1.errorbar(x=range(scores_all_single_units.shape[1]), y=np.mean(scores_all_single_units, axis=0), yerr=np.std(scores_all_single_units, axis=0), linewidth=0.5, label='single unit', color='m')
# ax1.errorbar(x=range(scores_all_single_units.shape[1]), y=np.mean(scores_all_single_units[0:650, :], axis=0), yerr=np.std(scores_all_single_units[0:650, :], axis=0), linewidth=0.5, label='single unit - 1st layer', ls='--', color='m')
# ax1.errorbar(x=range(scores_all_single_units.shape[1]), y=np.mean(scores_all_single_units[650:1300, :], axis=0), yerr=np.std(scores_all_single_units[650:1300, :], axis=0), linewidth=0.5, label='single unit - 2nd layer', ls=':', color='m')

#### Cosmetics and save figures
path2figures, filename = os.path.split(args.output_file_name)
# Fig 1d
sentence = ['The', 'boy', 'near', 'the', 'cars', 'greets']
ax1.set_xlim((0, len(sentence)-1))
ax1.axhline(0.5, color='k', ls = '--')
ax1.set_xticklabels(sentence, fontsize=40)
ax1.tick_params(axis='x', which='major', pad=15)
ax1.set_ylabel('Singular vs. Plural (AUC)', fontsize = 40)
ax1.set_yticks([0, 0.5, 1])
ax1.set_yticklabels([0, 0.5, 1], fontsize=30)
ax1.set_ylim((0, 1.05))
handles, labels = ax1.get_legend_handles_labels()
a, b = labels.index('Full-model minus LR-units'), labels.index('Unit 988 (LR)')
labels[b], labels[a] = labels[a], labels[b]
handles[b], handles[a] = handles[a], handles[b]
ax1.legend(handles, labels, loc=3, fontsize=35)#, bbox_to_anchor=(1.05, 1))
fig1.savefig(os.path.join(path2figures, 'GAT1d_' + args.gate + '_' + filename))

print('Figures were saved to: ' + os.path.join(path2figures, 'GAT1d_' + args.gate + '_' + filename))

# save2dict = {'scores_full_model': scores_full_model,
#              'scores_2nd_layer': scores_2nd_layer,
#              'scores_all_single_units': scores_all_single_units,
#              'LG_units': LG_units
#               }

save2dict = {'scores_full_model': scores_full_model,
             'scores_LR_units': scores_LR_units
             }

with open(os.path.join(path2figures, 'GAT1d_' + args.gate + '_' + filename[:-4] +'.pkl'), 'wb') as f:
    pickle.dump(save2dict, f)

# Add to 1d figure

# ###### plot dist of scores
# fig2, ax2 = plt.subplots(1)
# bar_width = 0.1
# scores_single_unit_on_subject_number = np.asarray(scores_reduced_model_all[:, 1])
# scores_single_unit_on_second_noun_number = np.asarray(scores_reduced_model_all[:, 4])
# for u, unit_num in enumerate(list(set(range(650, 1300)) - set(number_units))):
#     jitter = np.random.random(1) * bar_width - 3 * bar_width / 4
#     ax3.scatter(1 + jitter, scores_single_unit_on_subject_number[u], s=3)
#     ax3.scatter(2 + jitter, scores_single_unit_on_second_noun_number[u], s=3)
# ax3.set_ylabel('AUC', fontsize=16)
# ax3.set_xlim((0.5, 2.5))
# ax3.set_ylim((-0.1, 1.1))
# ax3.set_xticks([1, 2])
# ax3.set_xticklabels(['Subject', '2nd noun'])
# path2figures, filename = os.path.split(args.output_file_name)
# fig3.savefig(os.path.join(path2figures, 'AUCdist_' + filename))
#
# print('Units with highest AUC on subject:')
# print([(i+650, j) for (i, j) in zip(np.transpose(np.argsort(np.negative(scores_single_unit_on_subject_number)))[0:20],
#                                 np.transpose(-np.sort(np.negative(scores_single_unit_on_subject_number)))[0:20])])
#
# print('Units with highest AUC on second noun but trained on subject:')
# print([(i + 650, j) for (i, j) in
#        zip(np.transpose(np.argsort(np.negative(scores_single_unit_on_second_noun_number)))[0:20],
#        np.transpose(-np.sort(np.negative(scores_single_unit_on_second_noun_number)))[0:20])])
