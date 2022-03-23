# Import the model and the tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Download the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-1B-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-1B-distill")

# Define sentence
sent = "what is chat bot"

# Tokenize the sent
tokens = tokenizer(sent, return_tensors="pt")

# Passing through the sent to the Blenderbot model
gen = model.generate(**tokens)

# Decoding the model generated output
tokenizer.decode(gen[0])
