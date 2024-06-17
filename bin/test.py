import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text = "Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling. The novels chronicle the lives of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at Hogwarts School of Witchcraft and Wizardry."
sentences = sent_tokenize(text)

print(sentences)
