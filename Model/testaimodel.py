import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load the trained model from the folder we saved it in
model_path = "OpenCensor"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()

# if the score is above this number we say the text is toxic
# example: score=0.45 and threshold=0.39  toxic
K_THRESHOLD = 0.39 

def predict(text):
    # convert the text to numbers the model can read
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # run the text through the model
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # turn the raw output into a score between 0 and 1
    # 0 = clean, 1 = toxic
    probs = torch.sigmoid(logits.squeeze())
    score = probs.item()
    
    # compare the score to the threshold and decide
    is_toxic = score >= K_THRESHOLD
    
    print(f"Text: {text}")
    print(f"Score: {score:.4f} | Toxic: {is_toxic}")
    print("-" * 30)


predict("אני אוהב אותך יא נסיך")
predict("לך תזדיין יא אפס")
predict("איזה יום יפה היום")
predict("חתיכת זבל מסריח")
predict("אתה מזכיר לי את דוד")
predict("יא מלך גבר")
predict("יא שרירי")
predict("אתה נראה כמו שחקן כדורגל")