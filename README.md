This repo contains:
- The complete set of stimuli together with utility scripts used in the experiments described in: ["The emergence of number and syntax units in LSTM language models" NAACL2019](https://arxiv.org/abs/1903.07435).
- Code and instructions required for repliacting the figures in the paper - see [Code](/Code/)

* datasets:
  * NA_tasks:  stimuli used in the Number-Agreement tasks.
  * Tree_depth: stimuli, parses and corresponding number of open nodes used for predicting syntactic tree depth from LSTM activity.

```bash
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

The pre-trained LSTM English language model together with the corresponding datasets used for their training and evaluation can be found [here](https://github.com/facebookresearch/colorlessgreenRNNs)
