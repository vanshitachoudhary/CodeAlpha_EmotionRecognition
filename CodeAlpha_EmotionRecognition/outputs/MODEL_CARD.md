# Model card: Speech emotion classifier

**Task:** classify a speech clip into 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise).
**Selected model:** SVM (RBF), trained on MFCC / delta-MFCC sequences (deep models) or MFCC statistics + prosody (classical models).
Educational demo.

## Intended use
Demonstrating an explainable speech-emotion pipeline. **Not** for hiring, surveillance, mental-health or any decision about people.

## Data
RAVDESS (24 actors, acted North-American English speech) and TESS (2 actresses, acted words). RAVDESS "calm" is merged into "neutral";
TESS "pleasant surprise" is mapped to "surprise". All clips are acted, not spontaneous.

## Evaluation
Test accuracy 0.533 [0.467, 0.592], macro-F1 0.516 [0.446, 0.573] on unseen speakers (n=240 clips).
Model selection used validation speakers only.

## Limitations
- Acted emotions in studio conditions; real conversational speech is far harder and the model will be less accurate.
- Very few speakers (24 + 2), mostly North-American English; accents, languages, ages and recording devices outside this are unseen.
- Emotion categories are culturally constructed and speakers express them differently; predictions are probabilities, not facts.
- Small test split: wide confidence intervals; differences between models can be within noise.


## Ethical considerations
Emotion recognition can be misused. Do not use it to evaluate people (interviews, employees, students) or for surveillance.
