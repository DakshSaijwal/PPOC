"""
PolitiPulse NLP Workshop - Week 3 Follow-Up Assignment
Sentiment Analysis with VADER

Student : Daksh Saijwal
Roll No : 240318

Requirements: pip install vaderSentiment
The printed output produced by this script is included as a comment block at
the end of the file.
"""

import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

comments = [
    "The New Roads Policy is AMAZING!!!",
    "i dont think the new tax rule helps anyone.",
    "The committee met on Tuesday to review the budget.",
    "Honestly, this healthcare reform is a disaster.",
    "Farmers finally got the support they deserve, great move!!",
]

# ============================================================
# QUESTION 1 - Preprocessing Function + VADER Scoring
# ============================================================
print("=" * 60)
print("QUESTION 1")
print("=" * 60)


# (a) preprocess: lowercasing, punctuation removal, tokenization
def preprocess(text):
    text = text.lower()                                               # lowercasing
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation removal
    tokens = text.split()                                             # tokenization
    return tokens


print("\n(a) Preprocessed token lists:")
for c in comments:
    print(f"  {c!r}")
    print(f"    -> {preprocess(c)}")

# (b) VADER compound scores on ORIGINAL (non-preprocessed) comments
print("\n(b) VADER compound scores (original comments):")
for c in comments:
    compound = analyzer.polarity_scores(c)["compound"]
    print(f"  {compound:+.4f}  |  {c}")

# ============================================================
# QUESTION 2 - Mini Approval-Rating Pipeline
# ============================================================
print("\n" + "=" * 60)
print("QUESTION 2")
print("=" * 60)


# (a)
def get_compound_scores(comments):
    return [analyzer.polarity_scores(c)["compound"] for c in comments]


# (b)
def average_score(scores):
    return sum(scores) / len(scores)


# (c)
def approval_rating(avg_score):
    if avg_score >= 0.5:
        return "Strongly Approve"
    elif avg_score >= 0.05:
        return "Approve"
    elif avg_score > -0.05:
        return "Neutral"
    elif avg_score > -0.5:
        return "Disapprove"
    else:
        return "Strongly Disapprove"


# (d) run full pipeline
scores = get_compound_scores(comments)
avg = average_score(scores)
label = approval_rating(avg)

print("\n(d) Full pipeline on the five comments:")
print(f"  Compound scores : {[round(s, 4) for s in scores]}")
print(f"  Average score   : {avg:.4f}")
print(f"  Approval rating : {label}")

# ============================================================
# QUESTION 3 - Per-Comment Sentiment Classifier
# ============================================================
print("\n" + "=" * 60)
print("QUESTION 3")
print("=" * 60)


def classify_sentiment(text):
    compound = analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


print("\nPer-comment classification:")
for c in comments:
    print(f"  {classify_sentiment(c):8s} | {c}")


# ============================================================
# PROGRAM OUTPUT (captured by running this script)
# ============================================================
# ============================================================
# QUESTION 1
# ============================================================
#
# (a) Preprocessed token lists:
#   'The New Roads Policy is AMAZING!!!'
#     -> ['the', 'new', 'roads', 'policy', 'is', 'amazing']
#   'i dont think the new tax rule helps anyone.'
#     -> ['i', 'dont', 'think', 'the', 'new', 'tax', 'rule', 'helps', 'anyone']
#   'The committee met on Tuesday to review the budget.'
#     -> ['the', 'committee', 'met', 'on', 'tuesday', 'to', 'review', 'the', 'budget']
#   'Honestly, this healthcare reform is a disaster.'
#     -> ['honestly', 'this', 'healthcare', 'reform', 'is', 'a', 'disaster']
#   'Farmers finally got the support they deserve, great move!!'
#     -> ['farmers', 'finally', 'got', 'the', 'support', 'they', 'deserve', 'great', 'move']
#
# (b) VADER compound scores (original comments):
#   +0.7513  |  The New Roads Policy is AMAZING!!!
#   +0.3818  |  i dont think the new tax rule helps anyone.
#   +0.0000  |  The committee met on Tuesday to review the budget.
#   -0.2732  |  Honestly, this healthcare reform is a disaster.
#   +0.8118  |  Farmers finally got the support they deserve, great move!!
#
# ============================================================
# QUESTION 2
# ============================================================
#
# (d) Full pipeline on the five comments:
#   Compound scores : [0.7513, 0.3818, 0.0, -0.2732, 0.8118]
#   Average score   : 0.3343
#   Approval rating : Approve
#
# ============================================================
# QUESTION 3
# ============================================================
#
# Per-comment classification:
#   Positive | The New Roads Policy is AMAZING!!!
#   Positive | i dont think the new tax rule helps anyone.
#   Neutral  | The committee met on Tuesday to review the budget.
#   Negative | Honestly, this healthcare reform is a disaster.
#   Positive | Farmers finally got the support they deserve, great move!!
