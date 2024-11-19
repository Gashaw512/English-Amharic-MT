import spacy
import coreferee

# Load the transformer model and coreferee
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')

# Sample text for testing
text = "The writer interviewed the manager to get his feedback. The manager was happy."

# Process the text
doc = nlp(text)

# Print coreference chains
doc._.coref_chains.print()

# Save ambiguous terms for professional roles without paired pronouns
from spacy.matcher import PhraseMatcher

# Define professional terms
professional_terms = ["manager", "engineer", "doctor", "teacher", "writer"]

# Create a matcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(term) for term in professional_terms]
matcher.add("PROFESSIONAL_TERMS", patterns)

# Find matches
matches = matcher(doc)

ambiguous_terms = []
for match_id, start, end in matches:
    span = doc[start:end]
    if not any(pronoun in span.sent.text for pronoun in ["he", "she", "his", "her"]):
        ambiguous_terms.append(span.text)

print("Ambiguous Terms:", ambiguous_terms)


