from transformers import pipeline

class DynamicSummarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def summarize(self, text):
        # Calculate dynamic length based on input
        input_length = len(text.split())
        if input_length < 50:
            return text
        max_length = min(int(input_length * 0.4), 500)
        min_length = min(int(input_length * 0.2), max_length)
        
        try:
            summary = self.summarizer(text, 
                                    max_length=max_length, 
                                    min_length=min_length, 
                                    do_sample=False)
            return summary[0]['summary_text']
        except:
            return text
