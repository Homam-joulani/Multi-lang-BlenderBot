from googletrans import Translator

class ChatBot():
    def __init__(self, tokenizer: str, model: str):
        """
        Parameters:
        tokenizer: <class 'transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer'>
        model: <class 'transformers.models.blenderbot.modeling_blenderbot.BlenderbotForConditionalGeneration'>
        """
        self.tokenizer = tokenizer
        self.model = model
        self.translator = Translator()

    def translate(self, text: str, lang: str = "en"):
        """
        Text translation

        Parameters:
        text: str
            The text to be translated
        lang: str, default='en'
            The language to be translated into

        Return: str
            Translated text
        """
        return self.translator.translate(text, dest=lang).text

    def chat(self, text: str, lang: str = "en"):
        """
        Generating text

        Parameters:
        text: str
            Text to be sent to ChatBot
        lang: str, default='en'
            The ChatBot Language

        Return: str
            ChatBot Response
        """
        if lang != "en":
            tranText = self.translate(text, lang="en")
            text = tranText
        # Convert text into tokens
        tokens = self.tokenizer(text, return_tensors="pt")
        gen = self.model.generate(**tokens)
        result = self.tokenizer.decode(gen[0]).replace("<s>", "").replace("</s>", "")
        return self.translate(result, lang=lang)
