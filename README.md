# SlovakBERT (base-sized model)
SlovakBERT pretrained model on Slovak language using a masked language modeling (MLM) objective. This model is case-sensitive: it makes a difference between slovensko and Slovensko.

## Model link:
https://huggingface.co/gerulata/slovakbert

## Intended uses & limitations
You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task.
**IMPORTANT**: The model was not trained on the “ and ” (direct quote) character -> so before tokenizing the text, it is advised to replace all “ and ” (direct quote marks) with a single "(double quote marks).

## How to save tokenizer & model locally:
```python
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('gerulata/slovakbert')
tokenizer.save_pretrained('./slovakbert')
model = RobertaModel.from_pretrained('gerulata/slovakbert')
model.save_pretrained('./slovakbert')
```

### How to use
You can use this model directly with a pipeline for masked language modeling:

```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='gerulata/slovakbert')
unmasker("Deti sa <mask> na ihrisku.")

[{'sequence': 'Deti sa hrali na ihrisku.',
  'score': 0.6355380415916443,
  'token': 5949,
  'token_str': ' hrali'},
 {'sequence': 'Deti sa hrajú na ihrisku.',
  'score': 0.14731724560260773,
  'token': 9081,
  'token_str': ' hrajú'},
 {'sequence': 'Deti sa zahrali na ihrisku.',
  'score': 0.05016357824206352,
  'token': 32553,
  'token_str': ' zahrali'},
 {'sequence': 'Deti sa stretli na ihrisku.',
  'score': 0.041727423667907715,
  'token': 5964,
  'token_str': ' stretli'},
 {'sequence': 'Deti sa učia na ihrisku.',
  'score': 0.01886524073779583,
  'token': 18099,
  'token_str': ' učia'}]
```

Here is how to use this model to get the features of a given text in PyTorch:
```python
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('gerulata/slovakbert')
model = RobertaModel.from_pretrained('gerulata/slovakbert')
text = "Text ktorý sa má embedovať."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```
and in TensorFlow:
```python
from transformers import RobertaTokenizer, TFRobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = TFRobertaModel.from_pretrained('roberta-base')
text = "Text ktorý sa má embedovať."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```
Or extract information from the model like this:
```python
from transformers import pipeline
unmasker = pipeline('fill-mask', model='gerulata/slovakbert')
unmasker("Slovenské národne povstanie sa uskutočnilo v roku <mask>.")

[{'sequence': 'Slovenske narodne povstanie sa uskutočnilo v roku 1944.',
  'score': 0.7383289933204651,
  'token': 16621,
  'token_str': ' 1944'},...]
```

# Training data
The SlovakBERT model was pretrained on these datasets:

- Wikipedia (326MB of text),
- OpenSubtitles (415MB of text),
- Oscar (4.6GB of text),
- Gerulata WebCrawl (12.7GB of text) ,
- Gerulata Monitoring (214 MB of text),
- blbec.online (4.5GB of text)

The text was then processed with the following steps:
- URL and email addresses were replaced with special tokens ("url", "email").
- Elongated interpunction was reduced (e.g. -- to -).
- Markdown syntax was deleted.
- All text content in braces f.g was eliminated to reduce the amount of markup and programming language text.

We segmented the resulting corpus into sentences and removed duplicates to get 181.6M unique sentences. In total, the final corpus has 19.35GB of text.

# Pretraining
The model was trained in **fairseq** on 4 x Nvidia A100 GPUs for 300K steps with a batch size of 512 and a sequence length of 512. The optimizer used is Adam with a learning rate of 5e-4, \\(\beta_{1} = 0.9\\), \\(\beta_{2} = 0.98\\) and \\(\epsilon = 1e-6\\), a weight decay of 0.01, dropout rate 0.1, learning rate warmup for 10k steps and linear decay of the learning rate after. We used 16-bit float precision.

## About us
<a href="https://www.gerulata.com/">
	<img width="300px" src="https://www.gerulata.com/images/gerulata-logo-blue.png">
</a>

Gerulata uses near real-time monitoring, advanced analytics and machine learning to help create a safer, more productive and enjoyable online environment for everyone.

### BibTeX entry and citation info
If you find our resource or paper is useful, please consider including the following citation in your paper.
- https://arxiv.org/abs/2109.15254

```
@misc{pikuliak2021slovakbert,
      title={SlovakBERT: Slovak Masked Language Model}, 
      author={Matúš Pikuliak and Štefan Grivalský and Martin Konôpka and Miroslav Blšták and Martin Tamajka and Viktor Bachratý and Marián Šimko and Pavol Balážik and Michal Trnka and Filip Uhlárik},
      year={2021},
      eprint={2109.15254},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```