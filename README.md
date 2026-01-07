docker run --gpus all -it   -v "$PWD":/workspace   -w /workspace   tmsgnn:cu124 bash

<pre>
Number of participants: 72
Number of states: 4
Number of graphs: 103248
Number of graphs per state (true distribution): 
 {'wake': 58412, 'N1': 20869, 'N2': 16084, 'N3': 7883}

=====

55 participants in training, 
17 participants in testing.

Total number of unique graphs in training set: 78870
Number of graphs per state (true distribution): 
 {'wake': 44194, 'N1': 18704, 'N2': 12025, 'N3': 3947}

Total number of graphs in testing set: 24378
Number of graphs per state (true distribution): 
 {'wake': 14218, 'N1': 2165, 'N2': 4059, 'N3': 3936}

=====

9 participants in testing set group 1, 
this group contains participants who experienced all 4 brain states.
8 participants in testing set group 2, 
this group contains participants who did not fall asleep.

Total number of unique graphs in testing set group 1: 12906
Number of graphs per state (true distribution): 
 {'wake': 2746, 'N1': 2165, 'N2': 4059, 'N3': 3936}

Total number of unique graphs in testing set group 2: 11472
Number of graphs per state (true distribution): 
{'wake': 11472}
</pre>