# This script parse Marco's sentences + metadata files into separate files: sentences, Theo's pkl meta file and
# Tal's meta file.

import argparse

parser = argparse.ArgumentParser(description='Switching from Marco files to Tal format')
parser.add_argument('-i', '--input', required=True, help='Input sentences in Marco\'s format')
parser.add_argument('-o', '--output', required=True, help='Filename (without extension) for Tal\'s format - only path and basename should be specified. The script will then generate three files with the following extensions: text (sentences), gold (labels)  and info (pickle in Theo\'s format), which can be then used for further analyses and ablation experiments.')
parser.add_argument('-c', '--correct-word-position', type=str, help='The position of the (correct) test word (verb or adj) in the sentence, counting from ZERO. For example, <the boy near the cars greets the> it should be set to 5 if prediction test is on the verb') 
parser.add_argument('-p', '--pattern', type=str, nargs=2, action='append', help='Info regarding where the is the field (counting from ZERO) in Marco metadata regarding the number of first noun, second noun,..., first verb form, second verb form, etc. This should be provided as pairs as in the following example: -p number_1 2 -p number_2 3 -p number_3 4 -p verb_1_wrong = 5 -p verb_2_wrong 6 -p verb_3_wrong 7.') 
parser.add_argument('-w', '--wrong-word-label', type=str, help='Label (/key) name of the column in the metadata as defined by args.pattern above. For example, for nounpp: verb_1_wrong, and for objrel_that: verb2_wrong.') 
args = parser.parse_args()

# Open raw text file from Marco's sentence generator
with open(args.input, 'r') as f:
    raw_sentences = f.readlines()

# Split each row to different fields and save separately the sentences and the info
sentences = [line.split('\t')[1]+ '\n' for line in raw_sentences]
# For Theo's format:
info = []
for line in raw_sentences:
    curr_info = {}
    curr_line = line.split('\t')
    curr_info['RC_type'] = curr_line[0]
    for key, value in args.pattern:
        #print(key, value)
        curr_info[key] = curr_line[int(value)]
    curr_info['sentence_length'] = len(curr_line)
    info.append(curr_info)


# Prepare in Tal's format:
num_attributes = '-999'
sentences_Tal = []; gold_Tal = []; info_Tal = []
for sentence, curr_info in zip(sentences, info):
   correct_word = sentence.split(' ')[int(args.correct_word_position)].strip()
   wrong_word = curr_info[args.wrong_word_label].strip()
   
   line_Tal = str(args.correct_word_position) + '\t' + correct_word + '\t' + wrong_word + '\t' + num_attributes + '\n'
   
   sentences_Tal.append(sentence)
   gold_Tal.append(line_Tal)
   info_Tal.append(curr_info)


import os, pickle
# Save sentences in a text file
with open(args.output + '.text', 'w') as f:
    f.writelines(sentences_Tal)
    print('Sentences, gold labels and info files were saved to: ' + os.path.dirname(args.output))

# Save meta for Tal's agreement task
with open(args.output + '.gold', 'w') as f:
    f.writelines(gold_Tal)

# Save info in Theo's format
with open(args.output + '.info', 'wb') as f:
    pickle.dump(info_Tal, f)



























# Filter certain sentence types if desired.
#if args.filter_sentences:
#    IX_to_keep = []
#    for sent_type in keep_sentences:
#        IX_to_keep.append([IX for IX, sent_info in enumerate(info) if
#                      sent_info['RC_type'] == sent_type[0] and
#                      sent_info['number_1'] == sent_type[1] and
#                      sent_info['number_2'] == sent_type[2]])
#    IX_to_keep = [IX for sublist in IX_to_keep for IX in sublist]
#    # Filter sentence and info
#    sentences = [curr_sentence for IX, curr_sentence in enumerate(sentences) if IX in IX_to_keep]
#    info = [curr_info for IX, curr_info in enumerate(info) if IX in IX_to_keep]



#arser.add_argument('-f', '--filter-sentences', action='store_true', default=False, help = 'whether to filter sentences according to specific stimulus types (e.g., only subjrel_that). If this argument is given then the user needs to change the \'hard-coded\' keep_sentences var at the top of the code.')
#if args.filter_sentences:
    # Sublists containing which types of sentences to keep in the output files
#    keep_sentences = [['subjrel_that', 'singular', 'singular'], ['subjrel_that', 'plural', 'singular'], ['subjrel_that', 'plural', 'plural'], ['objrel', 'plural', 'plural'], ['objrel_that', 'plural', 'plural']]

   #verb_2_position = None
   #if curr_info['RC_type'] == 'objrel':
   #    verb2_position = 5
   #    verb2_correct = sentence.split(' ')[verb2_position].strip()
   #    verb2_wrong = curr_info['verb_2_wrong'].strip()
   #elif curr_info['RC_type'] == 'objrel_that':
   #    verb2_position = 6
   #    verb2_correct = sentence.split(' ')[verb2_position].strip()
   #    verb2_wrong = curr_info['verb_2_wrong'].strip()
   #elif curr_info['RC_type'] == 'subjrel':
   #    verb2_position = 6
   #    verb2_correct = sentence.split(' ')[verb2_position].strip()
#       verb2_wrong = curr_info['verb_2_wrong'].strip()
#   elif curr_info['RC_type'] == 'subjrel_that':
#       verb2_position = 6
#       verb2_correct = sentence.split(' ')[verb2_position].strip()
#       verb2_wrong = curr_info['verb_2_wrong'].strip()
#   elif curr_info['RC_type'] == 'double_subjrel_that':
#       verb2_position = 10 # Note that in this case of double subjrel we typically test the network on the *third* (verb_3) and not second verb in the sentence (see two lines below).
#       verb2_correct = sentence.split(' ')[verb2_position].strip()
#       verb2_wrong = curr_info['verb_3_wrong'].strip()
#   elif curr_info['RC_type'] == 'nounpp':
#       verb2_position = 5 # We test on verb_1 (see two lines below)
#       verb2_correct = sentence.split(' ')[verb2_position].strip()
#       verb2_wrong = curr_info['verb_1_wrong'].strip()
#   elif curr_info['RC_type'] == 'nounpp_adv':
#       verb2_position = 6 # We test on verb_1 (see two lines below)
#       verb2_correct = sentence.split(' ')[verb2_position].strip()
#       verb2_wrong = curr_info['verb_1_wrong'].strip()
#   elif curr_info['RC_type'] == 'adv_conjunction':
#       verb2_position = 5 # We test on verb_1 (see two lines below)
#       verb2_correct = sentence.split(' ')[verb2_position].strip()
#       verb2_wrong = curr_info['verb_1_wrong'].strip()
#   else:
#       print('Empty labels: ' + curr_info['RC_type'])
#       verb2_position = '0'
#       verb2_correct = ''
#       verb2_wrong = ''
#
