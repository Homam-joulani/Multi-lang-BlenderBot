# Import the model and the tokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import streamlit as st

from MultiLangBot import ChatBot

# Download the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

bot = ChatBot(tokenizer=tokenizer, model=model)

# Set the title of the web app
st.title("Multi Languages BlenderBot")

# Create select box
buff, col, buff2 = st.columns([1,10,20])
langBox = col.selectbox(
     'Pick your language',
     ('English', 'French', 'Arabic'))
if langBox == "English":
    lang = "en"
elif langBox == "French":
    lang = "fr"
elif langBox == "Arabic":
    lang = "ar"

# Create text box for input text
question = st.text_input('Text',"hi bot")
# Generate text from the ChatBot
gen = bot.chat(question, lang=lang)
# Create output box
answer = st.code(gen, language="markdown")
