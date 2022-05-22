# Classification

Writing heuristics for classification is extremely simple. Those functions can consist of as little as 3 lines of Python. We're going to look further into how they work, and wo we make use of them at Kern.

Generally, we automatically preprocess texts using [`spaCy`](https://spacy.io/), as this gives you potentially valuable metadata about the structures of your text.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
record = {"headline": "Data-centric AI is rising at super-speed!"}
record["headline"] = nlp(record["headline"])
```

We're now going to look into the different types of labeling functions you can implement for classifications. For simplicity, we assume that our data looks as follows:

```python
record = {
    "sender": "johannes.hoetter@kern.ai",
    "mail": "Hey, check this out, this is some email!",
}
```

## Generic labeling functions
Ultimately, labeling functions follow a simple interface. They take as input some `record`, i.e., a Python dictionary, and `return` a label name. If we have the two labels `spam` and `ham`, a function could look as follows:

```python
def is_capslock(record):
    if record["mail"].text.isupper(): # we need to call .text because of spacy
        return "spam"
```

Which tells the later used weak supervision model that mails that are written in full capslock tend to be spam mails. We don't have to make any assumption about how this would look like for `ham` in that function - that is a really helfpul characteristic of labeling functions. You don't have to go for a 100% coverage.

## Regex expression matching
Another great way to build labeling functions is using regular expressions. They can be as easy as:

```python
import re

def contains_send_money(record):
    YOUR_REGEX = r"send(.*?)money"
    YOUR_ATTRIBUTE = "mail"
    if re.match(YOUR_REGEX, record[YOUR_ATTRIBUTE].text):
        return "spam"
```

If you have struggle writing your regular expressions, check out these two great resources:
- [Regex tutorial — A quick cheatsheet by examples](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285)
- [Regex cookbook — Top 10 Most wanted regex](https://medium.com/factory-mind/regex-cookbook-most-wanted-regex-aa721558c3c1)

## Lookup functions
Generally, you can also include some kind of list to iterate through. At Kern, we automatically generate those lists from entites you manually label in extraction tasks, and store them in variables of the `knowledge` module. Let's say we have a knowledge base called `known_senders`:

```python
import knowledge

def lkp_known_sender(record):
    YOUR_ATTRIBUTE = "sender"
    for known_sender in knowledge.known_senders:
        # knowledge.senders might look like this: ["johannes.hoetter@kern.ai", "henrik.wenck@kern.ai", ...]
        if known_sender.lower() in record[YOUR_ATTRIBUTE].lower():
            return "ham"
```

## Python libraries
For a defined list of libraries, you can also just import some module and use their logic for your labeling functions. For instance, applying the sentiment analysis of `TextBlob` can be a great start:

```python
from textblob import TextBlob

def textblob_sentiment(record):
    YOUR_ATTRIBUTE = "mail"
    YOUR_SENSITIVITY = 0.5
    if TextBlob(record[YOUR_ATTRIBUTE].text).sentiment.polarity < -YOUR_SENSITIVITY:
        return "spam"
```

## 3rd-party and legacy systems
If you have some model or 3rd-party application you want to call, you can just integrate them via `requests`. For instance, let's say we have our production model running at `https://myapp.example.com`:

```python
import requests

def call_production_model(record):
    response = requests.post("https://myapp.example.com", json=record["mail"].text)
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        return prediction
```

## Active learning and zero-shot classifications
In addition to custom labeling functions, you can also easily implement active learners and zero-shot models on your classification tasks. This way, you can also express heuristics without knowing the exact inner mechanics of some labeling distribution.

## Further ideas?
If you want to add some own labeling function templates, please let us know. We're happy to add them here!
