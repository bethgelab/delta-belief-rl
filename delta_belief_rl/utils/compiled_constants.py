import re

INVALID_CHARS = re.compile(r'[\x00-\x1f\x7f-\xa0\u2028\u2029]')

QUESTION_WORDS_LEMMATIZED = frozenset(['what', 'where', 'when', 'why', 'how', 'who', 'whom', 'whose', 'which', 'be', 'do', 'can', 'coul', 'woul', 'shoul', 'will', 'shall', 'may', 'might', 'have'])

QUESTION_WORDS_STEMMED = frozenset(['what', 'where', 'when', 'why', 'how', 'who', 'whom', 'whose', 'which', 'is', 'ar', 'wa', 'were', 'do', 'doe', 'did', 'can', 'could', 'would', 'should', 'will', 'shall', 'mai', 'might', 'have', 'ha', 'had'])
