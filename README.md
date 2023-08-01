# beam-search
```beam-search``` implements beam search for part-of-speech tagging. 

Args: 
* ```test_data```: has the format “instanceName goldClass f1 v1 f2 v2 ...”, where an instance corresponds to a word and goldClass is the word’s POS tag according to the gold standard (cf. ex/test.txt)
* ```boundary_file```: the format of boundary file is one number per line, which is the length of a sentence (cf. ex/boundary.txt); for instance, if the first line is 46, it means the first sentence in test data has 46 words.
* ```model_file```: a MaxEnt model in text format (cf. input/m1.txt).
* ```beam_size```: the max gap between the lg-prob of the best path and the lg-prob of kept path: that is, a kept path should satisfy [lg(prob) + beam size ≥ lg(max_prob)], where max_prob is the prob of the best path for the current position. lg is base-10 log.
* ```top_N```: when expanding a node in the beam search tree, choose only the topN POS tags for the given word based on P (y | x).
* ```top_K```: the max number of paths kept alive at each position after pruning.

Returns: 
* ```sys_output```: has the format “instanceName goldClass sysClass prob”, where instanceName and goldClass are copied from the test data, sysClass is the tag y for the word x according to the best tag sequence found by the beam search, and prob is P (y | x) (cf. ex/sys.txt). Note prob is NOT the probability of the whole tag sequence given the word sentence. It is the probability of the tag y given the word x.

To run: 
```
beamsearch_src/maxent.sh input/sec19_21.txt input/sec19_21.boundary input/m1.txt output/sys_output beam_size topN topK
```
