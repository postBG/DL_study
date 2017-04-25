from collections import Counter

def bag_of_words(text):
    # TODO: Implement bag of words
    bag = Counter()
    for word in text.split(' '):
        bag[word] += 1
    return bag

test_text = 'the quick brown fox jumps over the lazy dog'

print(bag_of_words(test_text))