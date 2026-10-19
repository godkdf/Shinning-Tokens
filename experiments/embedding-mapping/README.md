# Historical o200k / BLOOM embedding mapping

Evidence: an existing local exploratory report, preserved here without rerunning the original model-weight experiment. The source is GPT-OSS-20B input vectors with width **2,880**; the target is BLOOM-560M input vectors with width **1,024**. These are model embedding spaces, not interchangeable vectors merely because tokenizer pieces match.

The report records **32,077 fit**, **3,828 validation**, and **4,095 test** anchors, with **107,404** predicted candidate vectors. Selected ridge regularization: **0.01**. Test coordinate MSE: **0.0001068646**, versus target training-mean baseline **0.0001323813**; ratio **0.8072485**, or **19.28% lower MSE**. Mean cosine: **0.5687569**. Exact source/target revisions and the transformation are in `historical-report.json`.

Scope: input-embedding initialization only. No output-head adaptation, downstream training or generation-quality evaluation is established. The report alone does not prove absence of anchor-selection leakage; a new reproduction must audit splits. Core CLI mapping and averaging baselines have numerical tests on disjoint synthetic anchors; that is not a reproduction of this historical model experiment.
