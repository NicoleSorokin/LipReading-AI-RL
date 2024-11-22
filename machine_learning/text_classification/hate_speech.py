from transformers import pipeline

pipe = pipeline("text-classification", model="ctoraman/hate-speech-bert", framework="pt")

result = pipe("you're' killing it")

print(result)