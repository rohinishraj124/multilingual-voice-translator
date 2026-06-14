from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Download complete!")