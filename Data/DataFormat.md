Explaination of the data files:

Databases:

* AVEC is the AVEC2012 database (word-level)
* utt-AVEC is AVEC2012 data down-sampled to the utterance level
* IEMOCAP is the IEMOCAP database (utterance-level)

Format of the .csv files:

* In DIS-NV files, the columns are {FilledPause, Filler, Stutter, Laughter, AudibleBreath}
* In exDN files, the columns are {FilledPause, Filler, Stutter, Laughter, AudibleBreath, Corrections, TurnTakingTime, Prolongation}
* In emo files, the columns are {Arousal, Expectancy, Power, Valence} for AVEC and utt-AVEC, {Arousal, Power, Valence} for IEMOCAP.
* Original annotations on each emotion dimension are grouped into three categories {low, medium, high}
* In GP files, the columns are {log-pitch, loudness, HF500}
