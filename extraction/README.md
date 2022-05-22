# Extraction
Writing labeling functions for extractors can be a bit hard at first, as they are more complex than classification heuristics. Do make life easier, we apply [`spaCy`](https://spacy.io/) to texts before putting them to labeling functions. This not only helps us to tokenize our data, but we can also access great out-of-the-box information from the `spaCy`-pipeline.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
record = {"headline": "Data-centric AI is rising at super-speed!"}
record["headline"] = nlp(record["headline"])
```

In Kern, this step is done automatically, so you don't have to worry about it. 

We're now going to look into the different types of labeling functions you can implement for extractions.

## Generic labeling functions
To start, we'll look at vanilla labeling functions. They are simple to understand and build the outline for all other types of labeling functions in extraction. In general, you need to loop over your tokens, implement some identification logic, and yield a triplet consisting of the label name, the span start index, and the span end index. An example could look as follows:

```python
record = {"details": nlp("The electronics chain said the used iPhones, which were returned within 30 days of purchase, are priced at $149 for the model with 8 gigabytes of storage,  while the 16-gigabyte version is $249")}
# we're skipping the spacy processing here and in the following examples

def detect_money(record):
    for token in record["details"][1:]:
        if token.text[0].isdigit() and token.nbor(-1).is_currency:
            yield "MONEY", token.i - 1, token.i + 1
            
for span in detect_money(record):
    print(f'{record["details"][span[1]: span[2]]} -> {span[0]}')
```

Of course, this can become cumbersome to implement, so please take a look at the below templates.

## Regular expression matchers
If you don't want to worry about matching the right indices, and only want to provide a regular expression, the following template is perfect for you:

```python
import re
record = {"details": nlp("The electronics chain said the used iPhones, which were returned within 30 days of purchase, are priced at $149 for the model with 8 gigabytes of storage,  while the 16-gigabyte version is $249")}

def detect_money_regex(record):
    YOUR_REGEX = "\$[0-9]+" # choose any regex here
    YOUR_ATTRIBUTE = "details" # choose any available attribute here
    YOUR_LABEL = "MONEY"

    def regex_search(pattern, string):
        """
        some helper function to easily iterate over regex matches
        """
        prev_end = 0
        while True:
            match = re.search(pattern, string)
            if not match:
                break

            start, end = match.span()
            yield start + prev_end, end + prev_end

            prev_end += end
            string = string[end:]
            
    for start, end in regex_search(YOUR_REGEX, record[YOUR_ATTRIBUTE].text):
        span = record[YOUR_ATTRIBUTE].char_span(start, end, alignment_mode="expand")
        yield YOUR_LABEL, span.start, span.end

for span in detect_money_regex(record):
    print(f'{record["details"][span[1]: span[2]]} -> {span[0]}')
```

If you have struggle writing your regular expressions, check out these two great resources:
- [Regex tutorial — A quick cheatsheet by examples](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285)
- [Regex cookbook — Top 10 Most wanted regex](https://medium.com/factory-mind/regex-cookbook-most-wanted-regex-aa721558c3c1)

## Window search matchers
Alternatively, you might want to find matches based on cue words within a certain window size, e.g. if it is difficult to list the entity you want to tag, but it is easy to define it by surrounding terms. To do so, you can implement a window-based approach:

```python
record = {"details": nlp("Max Mustermann decided to join Kern AI, where he wants to build great software.")}

def window_cue_search(doc):
    YOUR_WINDOW = 4 # choose any window size here
    YOUR_LABEL = "PERSON"
    YOUR_ATTRIBUTE = "details" # choose any available attribute here
    LOOKUP_VALUES = ["join"] # this could also be coming from a knowledge base via `import knowledge`
    for chunk in record[YOUR_ATTRIBUTE].noun_chunks:
        left_bound = max(chunk.sent.start, chunk.start - (YOUR_WINDOW // 2) +1)
        right_bound = min(chunk.sent.end, chunk.end + (YOUR_WINDOW // 2) + 1)
        window_doc = record[YOUR_ATTRIBUTE][left_bound: right_bound]
        if any([term in window_doc.text for term in LOOKUP_VALUES]):
            yield YOUR_LABEL, chunk.start, chunk.end
        
for span in window_cue_search(record):
    print(f'{record["details"][span[1]: span[2]]} -> {span[0]}')
```

Window search matchers become super-powerful if you match them with dynamic knowledge bases within the Kern UI. We'll automatically update those values for you, so that you don't have to worry about curating them.

## Aspect matchers
This type of labeling function is a bit more niche, but still super valuable: matching aspects. To implement them, we can just integrate libraries such as `TextBlob` into our function:

```python
record = {"details" : nlp("It has a really great battery life, but I hate the window size...")}

from textblob import TextBlob
def aspect_matcher(doc):
    YOUR_ATTRIBUTE = "details"
    YOUR_WINDOW = 4 # choose any window size here
    YOUR_SENSITIVITY = 0.5 # choose any value between 0 and 1
    NEGATIVE_LABEL = "NEGATIVE"
    POSITIVE_LABEL = "POSITIVE"
    for chunk in record[YOUR_ATTRIBUTE].noun_chunks:
        left_bound = max(chunk.sent.start, chunk.start - (YOUR_WINDOW // 2) +1)
        right_bound = min(chunk.sent.end, chunk.end + (YOUR_WINDOW // 2) + 1)
        window_doc = record[YOUR_ATTRIBUTE][left_bound: right_bound]
        sentiment = TextBlob(window_doc.text).polarity
        if sentiment < -YOUR_SENSITIVITY:
            yield NEGATIVE_LABEL, chunk.start, chunk.end
        elif sentiment > YOUR_SENSITIVITY:
            yield POSITIVE_LABEL, chunk.start, chunk.end

for span in aspect_matcher(record):
    print(f'{record["details"][span[1]: span[2]]} -> {span[0]}')
```

## Gazetters (lookup matchers)
Gazetters are super helpful if you want to repeat some labeling based on lookup values. For instance, if you already have some database or knowledge base you want to integrate, they become really helpful:

```python
record = {"details": nlp("Max Mustermann decided to join Kern AI, where he wants to build great software.")}

def gazetter(record):
    YOUR_ATTRIBUTE = "details"
    YOUR_LABEL = "PERSON"
    LOOKUP_VALUES = ["Max"]
    for chunk in record[YOUR_ATTRIBUTE].noun_chunks:
        if any([chunk.text in trie or trie in chunk.text for trie in LOOKUP_VALUES]):
            yield YOUR_LABEL, chunk.start, chunk.end

for span in gazetter(record):
    print(f'{record["details"][span[1]: span[2]]} -> {span[0]}')
```

Depending on your implementation, gazetters can really help to get label consistency throughout your documents. For instance, if you want to label that `Max Meier` is a person, a gazetter will ensure that you also find `Max` or `Mr. Meier` in your data.

## Further ideas?
If you want to add some own labeling function templates, please let us know. We're happy to add them here!
