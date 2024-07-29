RETRIEVAL_PROMPT_TPL = '''
Based on the retrieved examples of sentences and the relation classes between the head entity and the tail entity, please determine the relation class for the query sentence. The answer should be one of the relation classes provided in the examples.

----------
Retrieved Examples: 
{retrieved_examples}
----------

Query: 
{query_sentence}
What is the relation between head entity and tail entity?

'''

QUERY_PROMPT_TPL = '''
relationship between {entity1} and {entity2}
'''