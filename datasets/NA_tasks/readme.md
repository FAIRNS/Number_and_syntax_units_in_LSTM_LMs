# Presenting the Number Agreement data

Every file in this folder corresponds to one category of stimuli used in the Number Agreement (NA) task presented in the paper. 
They contains 4 columns, separated by tabulations:
- The first one contains the sentence
- The second one contains the plurality of the subject (singular or plural), 
  or, in the relative clause conditions, the plurality of the subject followed by that of the subject inside the relative clause (the distractor), 
  eg: singular_plural and plural_singular are the incongruent conditions described in the paper.
- The third column contains "correct" or "wrong", depending on whether this sentence has a correct number agreement or not.
  Each of the sentences is present twice, once with the correct agreement and once with the wrong one.
  The sentence is considered correct if the model assigns a higher probability to the correct verb than to the wrong one.
- The fourth column contains the sentence id (it is the same for both correct and wrong version of the same sentence).

The full ablation pipeline will be provided upon publication.
