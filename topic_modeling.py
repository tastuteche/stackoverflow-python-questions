from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def elapsed_timer():
    start = default_timer()

    # elapser = lambda: default_timer() - start
    last = None
    curr = start

    def elapser():
        nonlocal last, curr
        last = curr
        curr = default_timer()
        return (curr - last, curr - start)
    yield lambda: elapser()

    end = default_timer()

    def elapser(): return (end - start, end - start)


import pandas
print("""
# # Read in the questions and tags
# # Here I use 1/1000 question because of memory/time constraints on Kaggle
# # This leads to slightly different word/tag combinations in the topics
# # than in the blog post""")
with elapsed_timer() as elapsed:
    questions = pandas.read_csv(
        "./pythonquestions/Questions.csv", encoding='latin1').query('Id % 1000 == 0')
    print("Read in the questions at (%.2f, %.2f) seconds" % elapsed())
    tags = pandas.read_csv("./pythonquestions/Tags.csv", encoding='latin1')
    print("Read in the tags at (%.2f, %.2f) seconds" % elapsed())
print("Read in the questions and tags at (%.2f, %.2f) seconds" % elapsed())


import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(list(string.punctuation))
# stop_words.update(["gt", "lt"])

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# translator = str.maketrans('', '', string.punctuation)
replace_punctuation = str.maketrans(
    string.punctuation + '0123456789', ' ' * (len(string.punctuation) + 10))


print("")
with elapsed_timer() as elapsed:
    # questions['Body'] = questions["Body"].str.replace(r'<[^>]*>', '')
    import html2text
    questions['Body'] = questions["Body"].apply(
        lambda x: html2text.html2text(x))
    print("Body html2text at  (%.2f, %.2f) seconds" % elapsed())
    from nltk import word_tokenize
    questions['Body'] = questions['Body'].apply(
        lambda x: [stemmer.stem(word) for word in word_tokenize(x.lower().translate(replace_punctuation)) if word not in stop_words])
    #    lambda x: ' '.join([word for word in wordpunct_tokenize(x) if word not in stop_words]))
    print("Body tokenize stem  at  (%.2f, %.2f) seconds" % elapsed())
print("Body stem  at (%.2f, %.2f) seconds" % elapsed())


from gensim import corpora
print("""
# # Make a document-term matrix from the questions""")
with elapsed_timer() as elapsed:
    dictionary = corpora.Dictionary(questions["Body"])
    dictionary.filter_extremes(no_below=1, no_above=1, keep_n=5000)
    print("Body dictionary  at  (%.2f, %.2f) seconds" % elapsed())
    questions["bow"] = questions["Body"].map(dictionary.doc2bow)
    print("Body bow  at  (%.2f, %.2f) seconds" % elapsed())
print("Body dictionary bow  at (%.2f, %.2f) seconds" % elapsed())


import gensim
print("""
# # Fit a topic model (the time-consuming part)""")
with elapsed_timer() as elapsed:
    ldamodel = gensim.models.ldamodel.LdaModel(
        questions["bow"], num_topics=12, id2word=dictionary, passes=20)
print("ldamodel  at (%.2f, %.2f) seconds" % elapsed())

print("""
# # What are the top words for each topic?""")
for topic in ldamodel.print_topics(num_topics=12, num_words=10):
    print(topic)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd

import numpy as np


number = 12
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, number)]

fig, axes = plt.subplots(nrows=3, ncols=4)
fig.set_figheight(9)
fig.set_figwidth(12)
fig.tight_layout()
plt.title('Top terms in each LDA topic')
for ti, topic in enumerate(ldamodel.print_topics(num_topics=12, num_words=10)):
    print(topic)
    lst = {s1[1].replace(" ", '').replace('"', ''): float(s1[0].replace(
        ' ', '')) for s in topic[1].split('+') for s1 in [s.split('*')]}
    s = pd.Series(lst, name='word')
    ax = axes[ti // 4, ti % 4]
    ax.set_title("topic %s" % str(ti + 1))
    s.sort_values().plot('barh', ax=ax, color=colors[ti])


plt.savefig('top_terms_topic.png', dpi=200)
plt.clf()
plt.cla()
plt.close()

# # What are the top tags for each topic?
