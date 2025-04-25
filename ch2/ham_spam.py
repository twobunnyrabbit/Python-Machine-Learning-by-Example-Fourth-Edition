import pandas as pd
from collections import Counter
import itertools
from pyhere import here

sms_spam = pd.read_csv(here("data", "ch02", "spam", "SMSSpamCollection"), sep="\t", header=None)
sms_spam.columns = ["label", "sms"]
sms_spam.head()

sms_spam["sms"] = sms_spam["sms"].str.lower().str.split()
sms_spam.head()

sms_spam.loc[0:10, "sms"].apply(lambda x: [w.strip() for w in x])
sms_spam["sms"] = sms_spam["sms"].apply(lambda x: [w.strip() for w in x])
sms_spam.head()

# --- Functional Approach ---

# 1. Flatten the list of lists into a single iterator of words
#    itertools.chain.from_iterable is efficient for this.
# all_words_iterator = itertools.chain.from_iterable(df['sms'])
all_words_iterator = itertools.chain.from_iterable(sms_spam['sms'])

# 2. Count the frequency of each word across the entire dataset
#    collections.Counter is highly optimized for counting hashable objects.
total_word_counts = Counter(all_words_iterator)
type(total_word_counts)

# View the first few elements

first_few_elements = dict(itertools.islice(total_word_counts.items(), 10))
first_few_elements

# 3. Create the new DataFrame structure
#    - Get the number of rows for repeating the counts.
#    - Use a dictionary comprehension to create the data for the new DataFrame.
#      Each key is a word (column name), and each value is a list containing
#      the total count repeated for every row.
num_rows = len(sms_spam)
word_count_data = {
    word: [count] * num_rows
    for word, count in total_word_counts.items()
}
 