We want to leverage the v100s available on our cluster.
Consider the following and develop a plan:
* are there issues w/ data types (e.g., bf16, fp32, etc). How can we mitigate these issues?
* how can we clearly identify and track when things were run on a v100 vs other cards (or track all cards models were trained on)
* which models can fit on a v100 and which can't? how do we need to modify batch size or lora parameters?
* what performance issues can we expect? How much difference should we expect in bleu scores? how can we verify these ranges?
* how do we make sure the repo stays clean and well organized while allowing for v100 selection? We don't want flags embedded all over the place, we want a concise, elegant solution.