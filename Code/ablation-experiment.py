#!/usr/bin/env python
import sys
import os
import torch
import argparse
# sys.path.append(os.path.abspath('../src/word_language_model'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
    '../src/word_language_model')))
import data
import numpy as np
import h5py
import pickle
import pandas
import time
import copy
from tqdm import tqdm
from torch.autograd import Variable

parser = argparse.ArgumentParser(
    description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('model', type=str, default='model.pt',
                    help='Meta file stored once finished training the corpus')
parser.add_argument('-i', '--input', required=True,
                    help='Input sentences in Tal\'s format')
parser.add_argument('-v', '--vocabulary', default='reduced_vocab.txt')
parser.add_argument('-o', '--output', help='Destination for the output vectors')
parser.add_argument('--perplexity', action='store_true', default=False)
parser.add_argument('--eos-separator', default='</s>')
parser.add_argument('--fixed-length-arrays', action='store_true', default=False,
        help='Save the result to a single fixed-length array')
parser.add_argument('--format', default='npz', choices=['npz', 'hdf5', 'pkl'])
parser.add_argument('-u', '--unit', type=int, action='append', help='Which test unit to ablate')
parser.add_argument('-uf', '--unit-from', type=int, default=False, help='Starting range for test unit to ablate')
parser.add_argument('-ut', '--unit-to', type=int, default=False, help='Ending range for test unit to ablate')
parser.add_argument('-s', '--seed', default=1, help='Random seed when adding random units')
parser.add_argument('-g', '--groupsize', default=1, help='Group size of units to ablate, including test unit and random ones')
parser.add_argument('--unk-token', default='<unk>')
parser.add_argument('--use-unk', action='store_true', default=False)
parser.add_argument('--lang', default='en')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--do-ablation', action='store_true', default=False)
args = parser.parse_args()

stime = time.time()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Vocabulary
vocab = data.Dictionary(args.vocabulary)

# Sentences
sentences = [l.rstrip('\n').split(' ') for l in open(args.input + '.text', encoding='utf-8')]
gold = pandas.read_csv(args.input + '.gold', sep='\t', header=None, names=['verb_pos', 'correct', 'wrong', 'nattr'])

# Load model
print('Loading models...')
import lstm
print('\nmodel: ' + args.model+'\n')
model = torch.load(args.model, map_location=lambda storage, loc: storage)  # requires GPU model
model.rnn.flatten_parameters()
# hack the forward function to send an extra argument containing the model parameters
model.rnn.forward = lambda input, hidden: lstm.forward(model.rnn, input, hidden)
model_orig_state = copy.deepcopy(model.state_dict())

log_p_targets_correct = np.zeros((len(sentences), 1))
log_p_targets_wrong = np.zeros((len(sentences), 1))

# Compare performamce w/o killing units (set to zero the corresponding weights in model):
if args.unit_from and args.unit_to:
    if args.unit_to >= args.unit_from:
        target_units = [[x] for x in range(args.unit_from, args.unit_to+1)]
    else:
        target_units = [[x] for x in range(args.unit_from-1, args.unit_to-1, -1)]
else:
    target_units = [args.unit]

def feed_input(model, hidden, w):
    inp = torch.autograd.Variable(torch.LongTensor([[vocab.word2idx[w]]]))
    if args.cuda:
        inp = inp.cuda()
    out, hidden = model(inp, hidden)
    return out, hidden
def feed_sentence(model, h, sentence):
    outs = []
    for w in sentence:
        out, h = feed_input(model, h, w)
        outs.append(torch.nn.functional.log_softmax(out[0]).unsqueeze(0))
    return outs, h

for unit_group in tqdm(target_units):
    # restore the model to its original state in case it was ablated
    model.load_state_dict(model_orig_state)
    stime = time.time()
    # Which unit to kill + a random subset of g-1 more units
    np.random.seed(int(args.seed))
#    add_random_subset = np.random.permutation(1301).astype(int)
    if unit_group:
        units_to_kill = unit_group
    else:
#        add_random_subset = np.random.permutation(651).astype(int) + 650
#    add_random_subset = [i for i in add_random_subset if i not in unit_group] # omit current test unit from random set
        add_random_subset = np.random.permutation(1301).astype(int)
        units_to_kill = add_random_subset[0:(int(args.groupsize))] # add g random units
        unit_group = ['CONTROL'] + list(units_to_kill)

    units_to_kill = [u-1 for u in units_to_kill] # Change counting to zero
    units_to_kill_l0 = torch.LongTensor(np.array([u for u in units_to_kill if u <650])) # units 1-650 (0-649) in layer 0 (l0)
    units_to_kill_l1 = torch.LongTensor(np.array([u-650 for u in units_to_kill if u >649])) # units 651-1300 (650-1299) in layer 1 (l1)
    if args.cuda:
        units_to_kill_l0 = units_to_kill_l0.cuda()
        units_to_kill_l1 = units_to_kill_l1.cuda()

    output_fn = args.output
    if args.do_ablation: # if ablation then add unit number etc to filename
        output_fn = output_fn + "_".join(map(str, unit_group)) + '_groupsize_' + args.groupsize + '_seed_' + str(args.seed) # Update output file name
    output_fn = output_fn + '.abl'

    print("\n\n\n")
    print("ablated units: ", units_to_kill_l0.cpu().numpy() + 1, units_to_kill_l1.cpu().numpy() + 651)
    print("\n\n\n")

    for ablation in [args.do_ablation]: #[False, True]:
        if ablation:
            # Kill corresponding weights if list is not empty
            if len(units_to_kill_l0)>0: model.rnn.weight_hh_l0.data[:, units_to_kill_l0] = 0 # l0: w_hi, w_hf, w_hc, w_ho
            if len(units_to_kill_l1)>0: model.rnn.weight_hh_l1.data[:, units_to_kill_l1] = 0 # l0: w_hi, w_hf, w_hc, w_ho
            # if len(units_to_kill_l0)>0: model.rnn.weight_ih_l0.data[:, units_to_kill_l0] = 0 # l1: w_ii, w_if, w_ic, w_io
            # if len(units_to_kill_l1)>0: model.rnn.weight_ih_l1.data[:, units_to_kill_l1] = 0 # l1: w_ii, w_if, w_ic, w_io
            # if len(units_to_kill_l0)>0: model.rnn.bias_hh_l0.data[units_to_kill_l0] = 0
            # if len(units_to_kill_l1)>0: model.rnn.bias_hh_l1.data[units_to_kill_l1] = 0
            # if len(units_to_kill_l0)>0: model.rnn.bias_ih_l0.data[units_to_kill_l0] = 0
            # if len(units_to_kill_l1)>0: model.rnn.bias_ih_l1.data[units_to_kill_l1] = 0
            if len(units_to_kill_l1)>0: model.decoder.weight.data[:, units_to_kill_l1] = 0

        if args.lang == 'en':
            init_sentence = " ".join(["In service , the aircraft was operated by a crew of five and could accommodate either 30 paratroopers , 32 <unk> and 28 sitting casualties , or 50 fully equipped troops . <eos>",
                            "He even speculated that technical classes might some day be held \" for the better training of workmen in their several crafts and industries . <eos>",
                            "After the War of the Holy League in 1537 against the Ottoman Empire , a truce between Venice and the Ottomans was created in 1539 . <eos>",
                            "Moore says : \" Tony and I had a good <unk> and off-screen relationship , we are two very different people , but we did share a sense of humour \" . <eos>",
                            "<unk> is also the basis for online games sold through licensed lotteries . <eos>"])
        elif args.lang == 'it':
            init_sentence = " ".join(['Ma altre caratteristiche hanno fatto in modo che si <unk> ugualmente nel contesto della musica indiana ( anche di quella \" classica \" ) . <eos>',
            'Il principio di simpatia non viene abbandonato da Adam Smith nella redazione della " <unk> delle nazioni " , al contrario questo <unk> allo scambio e al mercato : il <unk> produce pane non per far- ne dono ( benevolenza ) , ma per vender- lo ( perseguimento del proprio interesse ) . <eos>'])

            #init_sentence = " ".join(["Si adottarono quindi nuove tecniche basate sulla rotazione pluriennale e sulla sostituzione del <unk> con pascoli per il bestiame , anche per ottener- ne <unk> naturale . <eos>", "Una parte di questa agricoltura tradizionale prende oggi il nome di agricoltura biologica , che costituisce comunque una nicchia di mercato di una certa rilevanza e presenta prezzi <unk> . <eos>", "L' effetto estetico non scaturisce quindi da un mero impatto visivo : ad esempio , nelle architetture riconducibili al Movimento Moderno , lo spazio viene modellato sulla base di precise esigenze funzionali e quindi il raggiungimento di un risultato estetico deriva dal perfetto adempimento di una funzione . <eos>"])
        else:
            raise NotImplementedError("No init sentences available for this language")

        if args.cuda:
            model = model.cuda()

        hidden = model.init_hidden(1) 
        init_out, init_h = feed_sentence(model, hidden, init_sentence.split(" "))

        # Test: present prefix sentences and calculate probability of target verb.
        for i, s in enumerate(tqdm(sentences)):
            out = None
            # reinit hidden
            #out = init_out[-1]
            hidden = init_h #model.init_hidden(1)
            # intitialize with end of sentence
            # inp = Variable(torch.LongTensor([[vocab.word2idx['<eos>']]]))
            # if args.cuda:
            #     inp = inp.cuda()
            # out, hidden = model(inp, hidden)
            # out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)
            for j, w in enumerate(s):
                if w not in vocab.word2idx and args.use_unk:
                    w = args.unk_token
                inp = Variable(torch.LongTensor([[vocab.word2idx[w]]]))
                if args.cuda:
                    inp = inp.cuda()
                out, hidden = model(inp, hidden)
                out = torch.nn.functional.log_softmax(out[0]).unsqueeze(0)
                if j==gold.loc[i,'verb_pos']-1:
                    assert s[j+1] == gold.loc[i, 'correct'].lower()
                    # Store surprisal of target word
                    log_p_targets_correct[i] = out[0, 0, vocab.word2idx[gold.loc[i,'correct']]].data.item()
                    log_p_targets_wrong[i] = out[0, 0, vocab.word2idx[gold.loc[i, 'wrong']]].data.item()
        # Score the performance of the model w/o ablation
        score_on_task = np.sum(log_p_targets_correct > log_p_targets_wrong)
        p_difference = np.exp(log_p_targets_correct) - np.exp(log_p_targets_wrong)
        score_on_task_p_difference = np.mean(p_difference)
        score_on_task_p_difference_std = np.std(p_difference)

        out = {
            'log_p_targets_correct': log_p_targets_correct,
            'log_p_targets_wrong': log_p_targets_wrong,
            'score_on_task': score_on_task,
            'accuracy_score_on_task': score_on_task,
            'sentences': sentences,
            'num_sentences': len(sentences),
            'nattr': list(gold.loc[:,'nattr']),
            'verb_pos': list(gold.loc[:, 'verb_pos'])
        }

        print(output_fn)
        print('\naccuracy: ' + str(100*score_on_task/len(sentences)) + '%\n')
        print('p_difference: %1.3f +- %1.3f' % (score_on_task_p_difference, score_on_task_p_difference_std))
        # Save to file
        if args.format == 'npz':
            np.savez(output, **out)
        elif args.format == 'hdf5':
            with h5py.File("{}.h5".format(output), "w") as hf:
                for k,v in out.items():
                    dset = hf.create_dataset(k, data=v)
        elif args.format == 'pkl':
            with open(output_fn, 'wb') as fout:
                pickle.dump(out, fout, -1)

