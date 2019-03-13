The complete set of stimuli together with utility scripts used in the experiments described in: ["The emergence of number and syntax units in LSTM language models" NAACL2019](add link to the paper).

Folder structure - 
* code:
* datasets:
  * NA_tasks:  stimuli used in the Number-Agreement tasks.
  * Tree_depth: stimuli, parses and corresponding number of open nodes used for predicting syntactic tree depth from LSTM activity.

```bash
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
```

Pre-trained LSTM models and datasets used for training and evaluating the LSTM language models can be found [here](https://github.com/facebookresearch/colorlessgreenRNNs)
