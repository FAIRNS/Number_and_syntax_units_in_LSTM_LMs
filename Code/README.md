# INSTRUCTIONS:
-------------
-------------

The steps described below will help you to replicate the figures presented in the [paper](https://arxiv.org/abs/1903.07435): 

Part 1 describes the scripts that: 
- Organize the stimuli in the required format.
- Extract gate and variable activations from the LSTM network and save them in a pkl file. 

Note that this part describes the required steps for the *Nounpp* number-agreement task (NA-task) only, but the step could be easily repeated for all other NA-tasks provided in the dataset folder. 

Part 2 describes the scripts required for regenerating figures 1-4 in the paper. Specifically,
- [plot_units_activations.py](plot_units_activations.py)
- [extract_embeddings_from_rnn.py](extract_embeddings_from_rnn.py)
- [extract_weights_from_rnn.py](extract_weights_from_rnn.py)



# PART 1 - data organization and extraction of LSTM activations:
-------------------------------------------------------------

data organization
-----------------
0. Prepare folders and Save model and its vocabulary:  
- After cloning the repository, you should create the following folders in your project: */Data/Stimuli/*, */Data/LSTM/activations/*, */Data/LSTM/models/*, */Output/* and */Figures/*.
- Before running the example below for the *Nounpp* task, make sure that the model and its vocabulary are saved in *Data/models/*. You can find on the [colorlessGreenRnns repo](https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/data) both the [english_model](https://dl.fbaipublicfiles.com/colorless-green-rnns/best-models/English/hidden650_batch128_dropout0.2_lr20.0.pt) and the [vocabulary](https://dl.fbaipublicfiles.com/colorless-green-rnns/training-data/English/vocab.txt).

1. Generate a dataset for the number-agreement task (NA-task):  
Instead of using one of the files in */Datasets/*, you could also generate data for an NA-task by using the script *generate_NA_task.pl*, for example, for the *Nounpp* task, run:
./Code/generate_NA_task.pl 0 1000 nounpp > Data/Stimuli/nounpp.txt

2. You can look into the statistics of the generated file by running:  
python Code/verify_stimuli_file_is_balanced.py -f Data/Stimuli/nounpp.txt > Data/Stimuli/nounpp.log

3. The result *nounpp.txt* should then be transformed into three files:
- \*.text: file containing only the sentence stimuli.
- \*.gold: file that contains for each sentence the correct and wrong predictions of the verb.
- \*.info: meta-data file with additional information.

To generate these three files, run:  
python Code/generate_info_from_raw_txt.py -i Data/Stimuli/nounpp.txt -o Data/Stimuli/nounpp -p number_1 2 -p number_2 3 -p verb_1_wrong 4 --correct-word-position 5 --wrong-word-label verb_1_wrong  
This will result in three files in Data/Stimuli/: nounpp.text, nounpp.gold and nounpp.info 

4. You can get the 'behavioral' performance of the LSTM on this NA-task, by running:  
python Code/extract_predictions.py Data/LSTM/models/hidden650_batch128_dropout0.2_lr20.0.pt -i Data/Stimuli/nounpp -v Data/LSTM/models/vocab.txt -o Output/nounpp --eos-separator "\<eos\>" --format pkl --lang en --uppercase-first-word

* Note that the pytorch model and vocab are assumed to be in *Data/LSTM/models/*
* you can use a GPU by adding a --cuda flag to the command.

Following this, run:  
python Code/get_agreement_accuracy_for_contrast.py -ablation-results Output/nounpp.abl -info Data/Stimuli/nounpp.info -condition number_1=singular number_2=singular  
python Code/get_agreement_accuracy_for_contrast.py -ablation-results Output/nounpp.abl -info Data/Stimuli/nounpp.info -condition number_1=singular number_2=plural  
python Code/get_agreement_accuracy_for_contrast.py -ablation-results Output/nounpp.abl -info Data/Stimuli/nounpp.info -condition number_1=plural number_2=singular  
python Code/get_agreement_accuracy_for_contrast.py -ablation-results Output/nounpp.abl -info Data/Stimuli/nounpp.info -condition number_1=plural number_2=plural  

5. Next, gate and unit activations should be extracted from the model. This is done by running:  
python Code/extract-activations.py Data/LSTM/models/hidden650_batch128_dropout0.2_lr20.0.pt -i Data/Stimuli/nounpp.text -v Data/LSTM/models/vocab.txt -o Data/LSTM/activations/nounpp --eos-separator "\<eos\>" --lang en --use-unk

6. Add model performance to metadata file:  
python Code/add_success_and_perplexity_to_info.py -i Data/Stimuli/nounpp.info -r Output/nounpp.abl -a Data/LSTM/activations/nounpp.pkl

# PART 2 - regenerate figures 1-4 in the paper:
----------------------------------------------

The scripts in this part require that:
- stimulus and metadata files for the *Nounpp* NA-task are in *Data/Stimuli/*: *nounpp.text* and *nounpp.info*, respectively.
- LSTM activations for the *Nounpp* NA-task are in *Data/LSTM/nounpp.pkl*.
- The LSTM model is saved in *Data/LSTM/models/hidden650_batch128_dropout0.2_lr20.0.pt* (the model can be downloaded from the [colorlessgreenRNNs repo](https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/data))

Launch the following commands from the root folder of the project and make sure that the paths specified in the arguments are indeed according to your local data organization.


FIGURE 1: dynamics of cell-suggestion, input and forget gates for units 776 and 988 and their efferent weights
--------------------------------------------------


### Unit 776 - nounpp
python [Code/plot_units_activations.py](plot_units_activations.py) -sentences Data/Stimuli/nounpp.text -meta Data/Stimuli/nounpp.info -activations Data/LSTM/activations/nounpp.pkl -o Figures/nounpp_775.png -c nounpp -g 4 r \- 6 775 cell number_1 singular number_2 singular success correct -g 1 r \- 6 775 gates.c_tilde number_1 singular number_2 singular success correct -g 1 r "\--" 6 775 gates.c_tilde number_1 singular number_2 plural success correct -g 1 b "\--" 6 775 gates.c_tilde number_1 plural number_2 singular success correct -g 1 b \- 6 775 gates.c_tilde number_1 plural number_2 plural success correct -g 2 r \- 6 775 gates.in number_1 singular number_2 singular success correct -g 2 r "\--" 6 775 gates.in number_1 singular number_2 plural success correct -g 2 b "\--" 6 775 gates.in number_1 plural number_2 singular success correct -g 2 b \- 6 775 gates.in number_1 plural number_2 plural success correct -g 3 r \- 6 775 gates.forget number_1 singular number_2 singular success correct -g 3 r "\--" 6 775 gates.forget number_1 singular number_2 plural success correct -g 3 b "\--" 6 775 gates.forget number_1 plural number_2 singular success correct -g 3 b \- 6 775 gates.forget number_1 plural number_2 plural success correct -g 4 r "\--" 6 775 cell number_1 singular number_2 plural success correct -g 4 b "\--" 6 775 cell number_1 plural number_2 singular success correct -g 4 b \- 6 775 cell number_1 plural number_2 plural success correct -g 4 g \- 6 1149 cell success correct -g 5 r \- 6 775 gates.out number_1 singular number_2 singular success correct -g 5 r "\--" 6 775 gates.out number_1 singular number_2 plural success correct -g 5 b "\--" 6 775 gates.out number_1 plural number_2 singular success correct -g 5 b \- 6 775 gates.out number_1 plural number_2 plural success correct -r 1 -x "The" "boy(s)" "near" "the" "car(s)" "greet(s)" "the" -y "$\tilde{C_t}$" "\$i\_t\$" "\$f\_t\$" "\$C\_t\$" "\$o\_t\$" --no-legend --facecolor w

![alt_text](https://github.com/FAIRNS/sentence-processing-MEG-LSTM/blob/master/Figures_paper/nounpp_775.png)

### Unit 988 - nounpp
python [Code/plot_units_activations.py](plot_units_activations.py) -sentences Data/Stimuli/nounpp.text -meta Data/Stimuli/nounpp.info -activations Data/LSTM/activations/nounpp.pkl -o Figures/nounpp_987.png -c nounpp -g 4 r \- 6 987 cell number_1 singular number_2 singular success correct -g 4 r "\--" 6 987 cell number_1 singular number_2 plural success correct -g 4 b "\--" 6 987 cell number_1 plural number_2 singular success correct -g 4 b \- 6 987 cell number_1 plural number_2 plural success correct -g 3 r \- 6 987 gates.forget number_1 singular number_2 singular success correct -g 3 r "\--" 6 987 gates.forget number_1 singular number_2 plural success correct -g 3 b "\--" 6 987 gates.forget number_1 plural number_2 singular success correct -g 3 b \- 6 987 gates.forget number_1 plural number_2 plural success correct -g 2 r \- 6 987 gates.in number_1 singular number_2 singular success correct -g 2 r "\--" 6 987 gates.in number_1 singular number_2 plural success correct -g 2 b "\--" 6 987 gates.in number_1 plural number_2 singular success correct -g 2 b \- 6 987 gates.in number_1 plural number_2 plural success correct -g 1 r \- 6 987 gates.c_tilde number_1 singular number_2 singular success correct -g 1 r "\--" 6 987 gates.c_tilde number_1 singular number_2 plural success correct -g 1 b "\--" 6 987 gates.c_tilde number_1 plural number_2 singular success correct -g 1 b \- 6 987 gates.c_tilde number_1 plural number_2 plural success correct -g 4 g \- 6 1149 cell success correct -g 5 r \- 6 987 gates.out number_1 singular number_2 singular success correct -g 5 r "\--" 6 987 gates.out number_1 singular number_2 plural success correct -g 5 b "\--" 6 987 gates.out number_1 plural number_2 singular success correct -g 5 b \- 6 987 gates.out number_1 plural number_2 plural success correct -r 1 -x "The" "boy(s)" "near" "the" "car(s)" "greet(s)" "the" -y "$\tilde{C_t}$" "\$i\_t\$" "\$f\_t\$" "\$C\_t\$" "\$o\_t\$" --no-legend --facecolor w

![alt_text](https://github.com/FAIRNS/sentence-processing-MEG-LSTM/blob/master/Figures_paper/nounpp_987.png)

### Effernt weights
python [Code/extract_embeddings_from_rnn.py](extract_embeddings_from_rnn.py) -model Data/LSTM/models/hidden650_batch128_dropout0.2_lr20.0.pt -v Data/LSTM/models/vocab.txt -i Data/Stimuli/singular_plural_verbs.txt -u 775 987 1149 650 1299 -c b r g k k

![alt_text](https://github.com/FAIRNS/sentence-processing-MEG-LSTM/blob/master/Figures_paper/weight_dists_verbs.png)

FIGURE 2: generalization across time (GAT)
--------------------------------------------------
This script requires having [MNE-python](https://www.martinos.org/mne/stable/install_mne_python.html) installed.


python [Code/SR_vs_LR_units.py](SR_vs_LR_units.py) -s Data/Stimuli/nounpp.text -m Data/Stimuli/nounpp.info -a Data/LSTM/activations/nounpp.pkl -g cell -o Figures/GAT1d_cell_.png

![alt_text](https://github.com/FAIRNS/sentence-processing-MEG-LSTM/blob/master/Figures_paper/GAT1d_cell_.png)

FIGURE 3: Cell activity of the syntax unit 1150
-----------------------------------------------------

python [Code/plot_units_activations.py](plot_units_activations.py) -sentences Data/Stimuli/nounpp.text -meta Data/Stimuli/nounpp.info -activations Data/LSTM/activations/nounpp.pkl -o Figures/nounpp_1149_cell.png -c nounpp -g 1 g \- 6 1149 cell -y "\$C_t$" -x "The" "boy" "near" "the" "car" "greets" "the" -r 1 --no-legend

![alt_text](https://github.com/FAIRNS/sentence-processing-MEG-LSTM/blob/master/Figures_paper/nounpp_1149_cell.png)

* note that to successfully generate all figures you should prepare the stimuli and activations also for the other NA-tasks (adv_conjunction, subjrel_that and double_subjrel_that) in the same way. Then, to get the rest of the sub-figures of Figure 3, run:

python Code/plot_units_activations.py -sentences Data/Stimuli/adv_conjunction.text -meta Data/Stimuli/adv_conjunction.info -activations Data/LSTM/activations/adv_conjunction.pkl -o Figures/adv_conjunction_1149_cell.png -c adv_conjunction -g 1 g \- 6 1149 cell -y "\$C_t$" -x "The" "boy" "gently" "and" "kindly" "greets" "the" -r 1 --no-legend

python Code/plot_units_activations.py -sentences Data/Stimuli/subjrel_that.text -meta Data/Stimuli/subjrel_that.info -activations Data/LSTM/activations/subjrel_that.pkl -o Figures/subjrel_that_1149_cell.png -c subjrel_that -g 1 g \- 6 1149 cell -y "\$C_t$" -x "The" "boy" "that" "watches" "the" "dog" "greets" "the" -r 1 --no-legend

python Code/plot_units_activations.py -sentences Data/Stimuli/double_subjrel_that.text -meta Data/Stimuli/double_subjrel_that.info -activations Data/LSTM/activations/double_subjrel_that.pkl -o Figures/double_subjrel_that_1149_cell.png -c double_subjrel_that -g 1 g \- 6 1149 cell -y "\$C_t$" -x "The" "boy" "that" "watches" "the" "dog" "that" "watches" "the" "cat" "greets" "the" -r 1 --no-legend




FIGURE 4: connectivity among the syntax (1150) and the LR-number units (776 and 988)
-----------------------------------------------------------------------------------------------

python [Code/extract_weights_from_rnn.py](extract_weights_from_rnn.py) -model Data/LSTM/models/hidden650_batch128_dropout0.2_lr20.0.pt -fu 775 987 1149 -tu 775 987 -o Figures/interactions.png --no-mds -activations Data/LSTM/activations/nounpp.pkl

![alt_text](https://github.com/FAIRNS/sentence-processing-MEG-LSTM/blob/master/Figures_paper/gate_Forget_afferent_interactions.png)


