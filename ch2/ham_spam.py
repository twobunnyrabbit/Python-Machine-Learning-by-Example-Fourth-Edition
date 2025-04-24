import pandas as pd
from collections import Counter
import itertools
from pyhere import here

sms_spam = pd.read_csv(here("data", "ch02", "spam", "SMSSpamCollection"), sep="\t", header=None)
sms_spam.columns = ["label", "sms"]
sms_spam.head()

sms_spam["sms"] = sms_spam["sms"].str.lower().str.split()
sms_spam.head()

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text if text != "" else None

sms_spam.loc[0:10, "sms"].apply(lambda x: [clean_text(w) for w in x])


