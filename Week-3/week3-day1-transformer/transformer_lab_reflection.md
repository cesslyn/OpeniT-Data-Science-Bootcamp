• Why do transformers need positional encoding?
Transformers need positional encoding because they don't process tokens in order by default. Positional encoding give the model information about the position of each token in the sequence.

• In your own words, what do queries, keys, and values do?
Queries look for relevant information, keys represent what each token offers, and values are the actual information that gets combined based on attention scores.

• Why is attention divided by the square root of the key dimension?
It is done so that it will be prevented the dot products from becoming too large, which could make softmax unstable and cause very small gradients.

• What does each row of an attention matrix represent?
Each row shows how one token distibutes its attention across all tokens in the sequence.

• Why might multiple attention heads be more useful than one attention head?
Multiple attention heads be more useful than one attention head because it allow the model to learn different tyoes of relationships at the same time instead of relying on a single pattern.

• What part of the lab was easiest to understand, and what part needs more practice?
The easiest part of the lab is the basic attention calcultion, while the hardest part and needs more pracics is the multi-head attention such as splitting, reshaping, and combining heads.