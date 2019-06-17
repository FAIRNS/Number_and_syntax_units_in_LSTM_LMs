import os, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')

parser = argparse.ArgumentParser(description='Visualize unit activations from an LSTM Language Model')
parser.add_argument('-sentences', '--stimuli-file-name', type=str, help='Path to text file containing the list of sentences to analyze')
parser.add_argument('-meta', '--stimuli-meta-data', type=str, help='The corresponding meta data of the sentences')
parser.add_argument('-activations', '--LSTM-file-name', type=str, help='The corresponding sentence (LSTM) activations')
parser.add_argument('-o', '--output-file-name', type=str, help='Path to output folder for figures')
parser.add_argument('-c', '--condition', type=str, help='Which condition to plot: RC, nounpp, etc.')
parser.add_argument('-g', '--graphs', nargs='+', action='append', type=str,
                    help='Specify a curve to be added to the figure with the following info: subplot-number, color, '
                         'line-style, line-width, unit number, gate, and key-value pairs as in Theo\'s meta info, '
                         'e.g., -g 1 b -- 775 forget number_1 singular number_2 singular '
                         '-g 2 g - 769 output number_1 plural number_2 singular.'
                         'gate should be one of: '
                         'gates.in, gates.forget, gates.out, gates.c_tilde, hidden, cell')
parser.add_argument('-r', '--remove', type=int, default=0, help='How many words to omit from the end of sentence')
parser.add_argument('-x', '--xlabels', nargs='+', type=str, help='List with xlabels for all subplors. Must match the number of time points')
parser.add_argument('-y', '--ylabels', nargs='+', type=str, help='List with ylabels for all subplors. Must match the number of subplots provided by --graphs')
parser.add_argument('--figwidth', type=int, default=15)
parser.add_argument('--figheight', type=int, default=15)
parser.add_argument('--xlabels-size', type=int, default=24)
parser.add_argument('--ylabels-size', type=int, default=24)
parser.add_argument('--yticks-size', type=int, default=16)
parser.add_argument('--no-legend', action='store_true', default=False, help='If specified, legend will be omitted')
parser.add_argument('--use-tex', default=False, action='store_true')
parser.add_argument('--facecolor', default=None)
args = parser.parse_args()

def get_unit_gate_and_indices_for_current_graph(graph, info, condition):
    '''

    :param graph: list containing info regarding the graph to plot - unit number, gate and constraints
    :param info: info objcet in Theo's format
    :param condition: sentence type (e.g., objrel, nounpp).
    :return:
    unit - unit number
    gate - gate type (gates.in, gates.forget, gates.out, gates.c_tilde, hidden, cell)
    IX_sentences - list of indices pointing to sentences numbers that meet the constraints specified in the arg graph
    '''
    color = graph[1]
    ls = graph[2]
    ls = ls.replace("\\", '')
    lw = float(graph[3])
    unit = int(graph[4])
    gate = graph[5]
    #print(len(graph))
    if len(graph) % 2 == 1:
        label = graph[-1]
    else:
        label = None
    # constraints are given in pairs such as 'number_1', 'singular' after unit number and gate
    keys_values = [(graph[i], graph[i + 1]) for i in range(6, len(graph)-1, 2)]
    if not label:
        label = str(unit) + '_' + gate + '_' + '_'.join([key + '_' + value for (key, value) in keys_values])
        if args.use_tex:
            label = label.replace("_", "-")
    IX_to_sentences = []
    for i, curr_info in enumerate(info):
        check_if_contraints_are_met = True
        for key_value in keys_values:
            key, value = key_value
            if curr_info[key] != value:
                check_if_contraints_are_met = False
        if check_if_contraints_are_met and curr_info['RC_type']==condition:
            IX_to_sentences.append(i)
    return unit, gate, IX_to_sentences, label, color, ls, lw

def add_graph_to_plot(ax, LSTM_activations, unit, gate, label, c, ls, lw):
    '''

    :param LSTM_activations(ndarray):  LSTM activations for current *gate*
    :param stimuli (list): sentences over which activations are averaged
    :param unit (int): unit number to plot
    :param gate (str): gate type (gates.in, gates.forget, gates.out, gates.c_tilde, hidden, cell)
    :return: None (only append curve on active figure)
    '''
    if LSTM_activations: print('Unit ' + str(unit))
    if gate.find('gate')==0: gate = gate[6::] # for legend, omit 'gates.' prefix in e.g. 'gates.forget'
    # Calc mean and std
    if LSTM_activations:
        mean_activity = np.mean(np.vstack([LSTM_activations[i][unit, :] for i in range(len(LSTM_activations))]), axis=0)
        std_activity = np.std(np.vstack([LSTM_activations[i][unit, :] for i in range(len(LSTM_activations))]), axis=0)

        # Add curve to plot
        ax.errorbar(range(1, mean_activity.shape[0] + 1), mean_activity, yerr=std_activity,
                label=label, ls=ls, lw=lw, color=c)
        offset = 0.15
        if gate in ['in', 'forget', 'out']:
            ax.set_yticks([0, 1])
            ax.set_ylim([0-offset, 1+offset])
        #elif gate in ['hidden']:
        #    ax.set_yticks([-0.05, 0.05])
        #    ax.set_ylim([-0.05-offset, 0.05+offset])
        #elif gate in ['cell', 'c_tilde']:
        #    ax.set_yticks([-3.5, 3.5])
        #    ax.set_ylim([-3.5-offset, 3.5+offset])
        #####ax.set_yticks(np.arange(min(-1, min(mean_activity)), 1+max(np.ceil(max(mean_activity)), 1), 1.0))
    else:
        print('No trials found for: ' + label)
if args.use_tex:
    plt.rc('text', usetex=True)

# make output dir in case it doesn't exist
os.makedirs(os.path.dirname(args.output_file_name), exist_ok=True)

###### Load LSTM activations, stimuli and meta-data ############
print('Loading pre-trained LSTM data...')
LSTM_activation = pickle.load(open(args.LSTM_file_name, 'rb'))
print('Loading stimuli and meta-data...')
with open(args.stimuli_file_name, 'r') as f:
    stimuli = f.readlines()
info = pickle.load(open(args.stimuli_meta_data, 'rb'))

##### Plot all curves on the same figure #########
subplot_numbers = [int(graph_info[0]) for graph_info in args.graphs]
num_subplots = np.max(subplot_numbers)
fig, axs = plt.subplots(num_subplots, 1, sharex=True, figsize=(args.figwidth,args.figheight),subplot_kw={'fc':args.facecolor})
if num_subplots==1: axs=[axs] # To make the rest compatible in case of a single subplot
for g, graph in enumerate(args.graphs):
    subplot_number = subplot_numbers[g]-1
    unit, gate, IX_to_sentences, label, color, ls, lw = get_unit_gate_and_indices_for_current_graph(graph, info, args.condition)
    if IX_to_sentences: print(gate, label)
    graph_activations = [sentence_matrix for ind, sentence_matrix in enumerate(LSTM_activation[gate]) if ind in IX_to_sentences]
    curr_stimuli = [sentence for ind, sentence in enumerate(stimuli) if ind in IX_to_sentences]
    if args.remove > 0:
        graph_activations = [sentence_matrix[:, 0:-args.remove] for sentence_matrix in graph_activations]
        curr_stimuli = curr_stimuli[0][0:-args.remove]
    # print('\n'.join(curr_stimuli))
    add_graph_to_plot(axs[subplot_number], graph_activations, unit, gate, label, color, ls, lw)

# Cosmetics
if graph_activations: axs[0].set_xticks(range(1, graph_activations[1].shape[1] + 1))
for i, ax in enumerate(axs):
    ax.grid(c='w', ls='-', lw=1)
    if args.xlabels:
        ax.set_xticklabels(args.xlabels, fontsize=args.xlabels_size)#, rotation='vertical')
    else:
        ax.set_xticklabels(stimuli[0].split(' '), fontsize=args.xlabels_size) #, rotation='vertical')
    #ax.tick_params(labelsize=10)
    ax.tick_params(axis='y', labelsize=args.yticks_size)
    if args.ylabels:
        ax.set_ylabel(args.ylabels[i], rotation='horizontal', ha='center', va='center', fontsize=args.ylabels_size)
    #else:
        #ax.set_ylabel('Activation', fontsize=45)
# adding legend
handles, labels = axs[0].get_legend_handles_labels() # take labels from cell, which also has a curve for the syntax unit 1150
#labels = ['Singular-Singular', 'Singular-Plural', 'Plural-Singular', 'Plural-Plural', 'Syntax unit 1150 (all conditions)']
legend = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10)
if args.no_legend: 
    legend.set_visible(False)

#fig.align_ylabels(axs)
#fig.align_ylabels(axs)
fig.align_ylabels(axs)
fig.align_ylabels(axs)

plt.tight_layout()
# Save and close figure
#plt.subplots_adjust(left=0.15, hspace=0.25)
plt.savefig(args.output_file_name)
plt.savefig(os.path.splitext(args.output_file_name)[0] +'.png') # Save also as svg
plt.close(fig)
print('The figure was saved to: ' + args.output_file_name)

