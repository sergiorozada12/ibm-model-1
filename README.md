# Statistical Machine Translation

This is a pretty dumb approach to train and use a really simple Statistical Machine Translation (SMT) pipeline. Don't take the approach too seriously, lot of things are far from being efficient, or directly wrong. The main components are:
* Translation model: IBM Model 1
* Language model: simple ngram statistical model without smoothing
* Decoder: greedy decoding

The corpus that I have used to train the model is Tatoeba ca-es.