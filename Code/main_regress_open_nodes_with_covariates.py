import os, sys, pickle, random
from functions import load_settings_params as lsp
from functions import model_fitting_and_evaluation as mfe
from functions import data_manip
from functions import annotated_data
from functions import vif
from functions import plot_results
from functions import prepare_for_ablation_exp
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../src/word_language_model'))

# inp
path = sys.path[0].split('/')
i = path.index('sentence-processing-MEG-LSTM')
base_folder = os.sep + os.path.join(*path[:i+1])

# number for filtering
n = 300
feature_type = 'hidden'
calc_VIF = False # VIF calc may be slow
omit_high_VIF_features = False

# base_folder = '/home/yl254115/Projects/FAIRNS'
txt_file = os.path.join(base_folder, 'Data/Stimuli/1M_sentences.txt')
model = os.path.join(base_folder, 'Data/LSTM/hidden650_batch128_dropout0.2_lr20.0.cpu.pt')
vocab = os.path.join(base_folder, 'Data/LSTM/english_vocab.txt')
data_file = os.path.join(base_folder, 'Output/num_open_nodes/activations_1M_sentences_n=%i.pkl'%n)
frequency_file = os.path.join(base_folder, 'Data/LSTM/english_word_frequencies.txt')

regenerate_data=False

eos = '<eos>'
use_unk = True
unk = '<unk>'
lang = 'en'
get_representations = ['word', 'lstm']
get_representations = ['lstm']
# out
print('Load settings and parameters')
settings = lsp.settings()
params = lsp.params()
preferences = lsp.preferences()
model_type = settings.method # Ridge/LASSO

base_filename = os.path.join(base_folder, 'Output/num_open_nodes/' + model_type + '_regression_number_of_open_nodes_n=%i'%n)


# Set random seeds:
np.random.seed(0)
random.seed(0)

### check if datafile exists, if not, create it, otherwise load it:
if os.path.exists(data_file) and not regenerate_data:
    print("Activations for this setting already generated, loading data from %s\n" % data_file)
    data_sentences = pickle.load(open(data_file, 'rb'))
else:
    data_sentences = annotated_data.Data()
    data_sentences.add_corpus(txt_file, separator='|', column_names=['sentence', 'structure', 'open_nodes_count', 'adjacent_boundary_count'])
    data_sentences.data = data_sentences.filter_sentences(n=n, elements=list(range(26))) # Filter data to get a uniform distribution of sentence types
    data_sentences.add_word_frequency_counts(frequency_file)
    data_sentences.add_activation_data(model, vocab, eos, unk, use_unk, lang, get_representations)
    pickle.dump(data_sentences, open(data_file, 'wb'))

### Decorrelate position and depth in data:
pos_min=7; pos_max=12; depth_min=3; depth_max=8

c_dict, plt = data_sentences.decorrelation_matrix(plot_pos_depth=True, pos_min=pos_min, pos_max=pos_max, depth_min=depth_min, depth_max=depth_max) # get position-depth tuples in data
plt.savefig(os.path.join(settings.path2figures, 'num_open_nodes', 'position_numON_plane.png'))
plt.close()
min_n = data_sentences.get_min_number_of_samples_in_rectangle(c_dict, pos_min=pos_min, pos_max=pos_max, depth_min=depth_min, depth_max=depth_max)

#data_sentences.decorrelate(pos_min=pos_min, pos_max=pos_max, depth_min=depth_min, depth_max=depth_max, n=min_n) # decorrelate data
#print('number of sentences after decorrelation = ', len(data_sentences.data))

_, filtered_data_full_dicts = data_sentences.decorrelate(pos_min=pos_min, pos_max=pos_max, depth_min=depth_min, depth_max=depth_max, n=min_n) # decorrelate data
print('number of sentences after decorrelation = ', len(data_sentences.data))
pickle.dump(data_sentences, open(data_file+'.dcl', 'wb'))


## write to a text file the full (non-filtered) dicts of the sentences after the decorrelation procedure
with open(data_file+'.txt', 'w') as f:
    for d in filtered_data_full_dicts:
        sentence = d['sentence']
        structure = d['structure']
        open_nodes_count = d['open_nodes_count']
        adjacent_boundary_count = d['adjacent_boundary_count']
        curr_line = '|'.join([sentence, structure, open_nodes_count, adjacent_boundary_count])
        f.write(curr_line+'\n')


# TODO implement function to find max rectangle
# TODO(?): data_sentences.omit_depth_zero() # Not needed for Marco's sentence generator

### Analyse features:
VIF = np.empty([0])
if calc_VIF:
    print('Calculating Variance Inflation Factor (VIF)')
    X, _ = data_manip.get_design_matrix(data_sentences.data, feature_type=feature_type)
    VIF = vif.calc_VIF(X)
    del X
    with open(base_filename + '_VIF.pkl', 'wb') as f:
        pickle.dump(VIF, f)

    # Thresh high VIF values --> features to omit from design matrix
    IX_valid_VIF = VIF <= 10

    # Log
    print('VIF: (min, max, mean) = (%1.2f, %1.2f, %1.2f)' % (np.min(VIF), np.max(VIF), np.mean(VIF)))
    print('Number of units after rejecting high-VIF units: %i' % IX_valid_VIF.sum())
    print('Rejected units: %s' % ' '.join([str(i) for i, v in enumerate(IX_valid_VIF) if not v]))

    # Plot VIF values hist:
    plt.hist(VIF, 50)
    plt.xlabel('VIF', size=18)
    plt.ylabel('Number of features', size=18)
    plt.savefig(os.path.join(settings.path2figures, 'num_open_nodes', 'VIF_dist.png'))
    plt.close()


# X, y, _, _ = data_manip.prepare_data_for_regression(data_sentences.data, data_sentences.data, feature_type=feature_type)
# print('number of positions for regression: ', y.shape[0])
### Train/test regression model:
print('Splitting train/test data')
models = []; weights = []; scores = []; scores_reduced_model = []; units_outliers = []
data_sentences_train, data_sentences_test = data_manip.split_data(data_sentences.data, params) # Train-test split
for split in range(params.CV_fold):
    # Preparing the data for regression, by breaking down sentences into X, y matrices that are word-wise:
    X_train, y_train, X_test, y_test = data_manip.prepare_data_for_regression(data_sentences_train[split], data_sentences_test[split], feature_type=feature_type)
    ### Plot scatter num-open-nodes vs. activations
    activations_1149 = [vec[1149] for vec in X_train]
    fig_scatter, ax_scatter = plt.subplots(1)
    # ax_scatter.scatter(y_train, activations_1149)
    x = []
    y = []
    yerr = []
    for open_n in range(max(y_train)):
        if open_n >= 1 and open_n < 12:
            x.append(open_n)
            y.append(np.mean(np.asarray(activations_1149)[y_train == open_n]))
            yerr.append(np.std(np.asarray(activations_1149)[y_train == open_n])/np.sqrt(np.sum(y_train == open_n)))
    ax_scatter.errorbar(x, y, yerr=yerr, color='k', lw=3)
    ax_scatter.set_xlabel('Number of open nodes', fontsize=20)
    ax_scatter.set_ylabel(feature_type.capitalize() + ' activity of unit 1149', fontsize=20)
    # ax_scatter.set_ylim([-0.1, 0.27])
    # ax_scatter.set_xlim([0, 12])
    fig_scatter.savefig(os.path.join(settings.path2figures, 'num_open_nodes', 'scatter_num_open_nodes_1149_' + feature_type + '.png'))
    plt.close(fig_scatter)

    X_train, X_test = data_manip.standardize_data(X_train, X_test)

    # Omit high-VIF units:
    if calc_VIF and omit_high_VIF_features:
        X_train = X_train[:, IX_valid_VIF]
        X_test = X_test[:, IX_valid_VIF]

    # Train a linear regression model:
    linear_model = mfe.train_model(X_train, y_train, settings, params)

    # Evaluate model (R-squared) on test set:
    scores_curr_split, MSE_per_depth = mfe.evaluate_model(linear_model, X_test, y_test, settings, params)
    scores.append(scores_curr_split)

    # Evalue reduced model (without covariats, such as word frequency):
    linear_model.best_estimator_.coef_[-1] = 0 # remove word-frequency regressor
    scores_reduced_model_curr_split, _ = mfe.evaluate_model(linear_model, X_test, y_test, settings, params)
    scores_reduced_model.append(scores_reduced_model_curr_split)

    # Save resulting weights
    curr_weights = linear_model.best_estimator_.coef_
    weights.append(curr_weights)
    models.append(linear_model)

    # Plot regularization path:
    p = plot_results.regularization_path(linear_model, settings, params)
    p.title('Split %i' % (split + 1))
    file_name = 'regularization_path_split_%i' % split + '.png'
    p.savefig(os.path.join(settings.path2figures, 'num_open_nodes', file_name))
    p.close()

    # For each split, find units with largest weights:
    num_features = curr_weights.shape[0]
    IX = np.abs(curr_weights).argsort()
    units_sorted = np.asarray(range(num_features))[IX[::-1]] # Sort units in descending order wrt weight size
    k, n, ave, std, IX_outliers = prepare_for_ablation_exp.get_weight_outliers(curr_weights) # Find outlier weights (>3SD)
    units_outliers.append(units_sorted[0:k]) # Append for each split

    # Print info to screen
    print('Split %i:' % (split+1))
    print('num_samples_train, num_samples_test, num_features = (%i, %i, %i)' % (X_train.shape[0], X_test.shape[0], X_train.shape[1]))
    print('k=%i, n=%i, mean_weight=%1.5f, std_weights=%1.5f' % (k, n, ave, std))
    print('Units for ablation: ' + ' '.join(['%i' % unit for unit in units_sorted[0:k]]))
    print('Mean validation score %1.2f +- %1.2f, alpha = %1.2f; Test scores: %1.5f (reduced model: %1.5f)' % (
        linear_model.cv_results_['mean_test_score'][linear_model.best_index_],
        linear_model.cv_results_['std_test_score'][linear_model.best_index_], linear_model.best_params_['alpha'],
        scores_curr_split, scores_reduced_model_curr_split))
    print('\n')

### Save to drive:
with open(base_filename + '.pkl', 'wb') as f:
    pickle.dump((models, scores, scores_reduced_model_curr_split, units_outliers), f)


### Plot stuff:
weights_mean = np.mean(np.asarray(weights), axis=0)
weights_std = np.std(np.asarray(weights), axis=0)
num_features = weights_mean.shape[0]
IX = np.abs(weights_mean).argsort()
units_sorted = np.asarray(range(num_features))[IX[::-1]]
k, n, ave, std, IX_outliers = prepare_for_ablation_exp.get_weight_outliers(weights_mean)

print('After averaging across splits:')
print('mean test score: %1.2f +- %1.2f' % (np.mean(scores), np.std(scores)))
print('k=%i, n=%i, mean_weight=%1.2f, std_weights=%1.2f' % (k, n, ave, std))
print('Units for ablation: ' + ' '.join(['%i' % unit for unit in units_sorted[0:k]]))

with open(os.path.join(settings.path2output, 'num_open_nodes', base_filename + '_weights.txt'), 'w') as f:
    for w in weights_mean[IX]: f.write("%f\n" % w)

plt.bar(range(num_features), weights_mean[IX], yerr=weights_std[IX])
plt.xlabel('Units (sorted by weight size)', size=18)
plt.ylabel('Weight size', size=18)
plt.savefig(os.path.join(settings.path2figures, 'num_open_nodes', 'weights_' + model_type + '_synthetic.png'))
plt.close()


fig1, ax1 = plt.subplots(1, figsize=[10,10])
N, bins, patches = ax1.hist(weights_mean, 50)
for i, bin in enumerate(bins):
    if bin < ave -3 *std or bin > 3 * std: patches[i-1].set_facecolor('r')
ax1.set_xlabel('Weight size', size=36)
ax1.set_ylabel('Number of units', size=36)
fig1.savefig(os.path.join(settings.path2figures, 'num_open_nodes', 'weights_' + model_type + '_synthetic_dist.png'))
plt.close(fig1)
