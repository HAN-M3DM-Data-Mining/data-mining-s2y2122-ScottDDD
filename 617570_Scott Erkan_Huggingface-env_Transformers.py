from transformers import pipeline

# results will be shown in the console 
# question-answering model
Scott_model1 = pipeline('question-answering')
question = 'What is my hobby?'
context = 'My name is Scott and my hobby is editing competitive music videos.'
Scott_model1(question = question, context = context)

#%%

# text summerization model 
classifier = pipeline('summarization')
classifier('Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.')

#%%

# text classification model
classifier = pipeline('text-classification', model = 'roberta-large-mnli')
classifier('A soccer game with multiple males playing. Some men are playing a sport.')

#%%

# text translation model
en_fr_translator = pipeline('translation_en_to_fr')
en_fr_translator('How old are you?')

