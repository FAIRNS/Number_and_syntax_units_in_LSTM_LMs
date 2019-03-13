The complete set of stimuli used in the experiments described in: ["The emergence of number and syntax units in LSTM language models" NAACL2019](add link to the paper).

Folder structure - 
* code: evaluation script.
* datasets:
  * NA_tasks:  datasets used for the Number-Agreement tasks,
  * Tree_depth: the dataset used for predicting the syntactic tree depth (number of open nodes) from LSTM activity.

Pre-trained LSTM models and datasets used for training and evaluating the LSTM language models can be found [here](https://github.com/facebookresearch/colorlessgreenRNNs)


'''
├── code
├── datasets
│   ├── NA_tasks
│   │   ├── adv_adv.txt
│   │   ├── adv_conjunction.txt
│   │   ├── adv.txt
│   │   ├── namepp.txt
│   │   ├── nounpp_adv.txt
│   │   ├── nounpp.txt
│   │   ├── readme.md
│   │   └── simple.txt
│   └── tree_depth
│       ├── filtered_sentences.txt
│       └── readme.md
├── README.md
└── supplementary_material.pdf

'''
