import pandas as pd
import torch
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tqdm import tqdm
# imports: csv reader, torch, scoring metrics, our custom dataset, nlp tools, progress bar

class CFG:
    KBaseModel = "onlplab/alephbert-base"
    KDataCsv = "data.csv"
    KMaxLen = 256 # max tokens per text
    KAutoLen = True # let the code figure out the best length from the data automatically
    KBatchSize = 16 # how many texts to process at once
    KEpochs = 10 # how many times we loop over the full dataset
    KLearningRate = 2e-5           
    KCheckpointFile = "censormodel.pt" # where we save the best weights
    KOutputDir = "censormodel" # folder for the final to be exported model at


def CleanText(s):
    # A filter example: hello! -> hello
    if pd.isna(s):
        return ""
    # Example: Removes emojis and fixes long letters (ןןןן -> ןן)
    text = " ".join(re.sub(r"[^\w @#$%*]+", " ", str(s)).split())
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) 
    return text


def AutoMaxLen(df, tokenizer):
    # This checks how long our sentences actually are so we dont waste computer memory
    # example: if most of our texts are just short tweets such as maybe 30 tokens it's silly to prepare massive 256 token boxes for them
    texts = df["text"].tolist()[:2000]

    if not texts:
        return 64

    lengths = []
    for text in texts:
        # we chop the text into tokens and turn them into numbers bc AI can only understand numbers
        # then we just count how many numbers (tokens) we got for each sentence
        tokens = tokenizer(text)["input_ids"]
        lengths.append(len(tokens))

    # we find a size that fits almost all of our sentences which is 95% of them
    p95 = int(np.percentile(lengths, 95))

    # we make sure the box isnt smaller than 64 tokens but not bigger than 256 tokens
    useLen = max(64, min(256, p95))
    return useLen

class TextDataset(Dataset):
    # example input: texts=["I hate you", "Hello"], labels=[1, 0], maxLen=5
    def __init__(self, texts, labels, tokenizer, maxLen):
        # saves ["I hate you", "Hello"] so we can access them later
        self.m_texts = texts
        # saves [1, 0] so we have the answers ready
        self.m_labels = labels
        # saves the AlephBERT dictionary tool that changes text to numbers
        self.m_tokenizer = tokenizer
        # saves 5 forcing every output to be exactly 5 tokens long
        self.m_maxLen = maxLen

    # pytorch asks how many items do we have to process in total?
    def __len__(self):
        # returns 2 since we have 2 sentences in our list
        return len(self.m_texts) 

    def __getitem__(self, idx):
        # example: text = str(["I hate you", "Hello"][0]) -> "I hate you"
        text = str(self.m_texts[idx])
        
        # example: we feed "I hate you" into the tokenizer to get numbers
        encoding = self.m_tokenizer(text, truncation=True, padding="max_length", max_length=self.m_maxLen, return_tensors="pt")
        return {
            "inputIds": encoding["input_ids"].squeeze(0), 
            "attentionMask": encoding["attention_mask"].squeeze(0), 
            "label": torch.tensor(self.m_labels[idx], dtype=torch.float32) 
            }

def GetPredictions(model, dataLoader, device):
    # we tell the AI to stop learning and just take the test
    # we also prep two blank lists to write down the guesses and the real answers
    model.eval()
    allProbs, allLabels = [], []

    # we turn off the AI is learning memory so the test runs faster
    with torch.no_grad(): 
        # the AI reads the sentences in small groups like 2 sentences at a time
        # example: it gets the numbers for "You are great" and "I hate you"
        for batch in dataLoader:
            inputIds = batch["inputIds"].to(device)
            attentionMask = batch["attentionMask"].to(device)
            
            # the AI looks at the text and spits out a score which we turn into a percentage
            # example: it guesses 11% toxic for the first and 95% toxic for the second -> [0.11, 0.95]
            outputs = model(input_ids=inputIds, attention_mask=attentionMask)
            probs = torch.sigmoid(outputs.logits.squeeze(-1)).cpu()

            # we save those guesses [0.11, 0.95] and the real answers [0, 1] to our lists
            allProbs.append(probs)
            allLabels.append(batch["label"])

    # finally we glue all the small groups of answers into one massive list for the whole test
    # final guesses: [0.11, 0.95, 0.18, 0.98] 
    # final real answers: [0, 1, 0, 1]
    probs = torch.cat(allProbs).numpy()
    labels = torch.cat(allLabels).int().numpy()
    
    return probs, labels


def Train():
    # example: forces PyTorch to use the graphics card
    device = torch.device("cuda")

    df = pd.read_csv(CFG.KDataCsv, quotechar='"', escapechar='\\', on_bad_lines='skip')
    
    # example: throws away any rows that have missing text or missing labels
    df.dropna(subset=["text", "label"], inplace=True)
    
    # example: runs every single sentence through our cleanText filter to fix spam letters
    df["text"] = df["text"].apply(CleanText)
    
    # before: "1" (text)
    # after: 1 (number)
    df["label"] = df["label"].astype(int)

    # example: we take your 100 sentences and split them into 3 piles:
    # pile 1 (Study): 80 sentences for the AI to learn from
    # pile 2 (Quiz): 10 sentences to test the AI while it studies
    # pile 3 (Final Exam): 10 sentences for the very last test
    
    # stratify=df["label"] makes sure each pile has the same mix of toxic and clean sentences
    trainDf, tempDf = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    valDf, testDf = train_test_split(tempDf, test_size=0.5, stratify=tempDf["label"], random_state=42)

    # example: this part gets the AI dictionary (tokenizer) and picks the best sentence length
    tokenizer = AutoTokenizer.from_pretrained(CFG.KBaseModel)
    if CFG.KAutoLen:
        CFG.KMaxLen = AutoMaxLen(trainDf, tokenizer)
        
    # example: this loads the actual AI brain and tells it we are doing 0 or 1
    model = AutoModelForSequenceClassification.from_pretrained(CFG.KBaseModel, num_labels=1).to(device)

    # example: this part takes your 3 piles of sentences and prepares them for the AI to read
    # it turns the words into numbers so the AI brain can understand them
    trainDataset = TextDataset(trainDf["text"].tolist(), trainDf["label"].tolist(), tokenizer, CFG.KMaxLen)
    valDataset = TextDataset(valDf["text"].tolist(), valDf["label"].tolist(), tokenizer, CFG.KMaxLen)
    testDataset = TextDataset(testDf["text"].tolist(), testDf["label"].tolist(), tokenizer, CFG.KMaxLen)

    trainLoader = DataLoader(trainDataset, batch_size=CFG.KBatchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=CFG.KBatchSize, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=CFG.KBatchSize, shuffle=False)

    # example: if we have way more clean sentences the AI might just guess clean every time to cheat
    # this weight acts like a penalty: it makes missing a toxic sentence a much bigger mistake
    posCount, negCount = (trainDf["label"] == 1).sum(), (trainDf["label"] == 0).sum()
    posWeight = torch.tensor([negCount / posCount]).to(device) if posCount > 0 else torch.tensor([1.0]).to(device)
    
    # example: the optimizer is the teacher it looks at the AI mistakes and reaches into its brain to turn the math knobs and fix them
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.KLearningRate)
    
    bestValF1 = -1.0
    
    # LOOP
    for epoch in range(1, CFG.KEpochs + 1):
        model.train()
        
        # we check 16 sentences at a time
        progressBar = tqdm(trainLoader, desc=f"Epoch {epoch}/{CFG.KEpochs}")
        for batch in progressBar:
            # we move the data to the GPU
            inputIds = batch["inputIds"].to(device)
            attentionMask = batch["attentionMask"].to(device)
            labels = batch["label"].to(device)
            
            # step 1: the AI looks at the text and makes a guess
            outputs = model(input_ids=inputIds, attention_mask=attentionMask)
            
            # step 2: we check how wrong the guess was.
            # high score = bad guess 
            # low score = good guess
            lossFn = torch.nn.BCEWithLogitsLoss(pos_weight=posWeight)
            loss = lossFn(outputs.logits.squeeze(-1), labels)
            
            # example: the AI fixes its brain in 3 steps:
            # 1. reset: it wipes the math from the last batch
            optimizer.zero_grad()  
            # 2. search: it finds which words caused the mistake
            loss.backward()        
            # 3. change: it turns the knobs to fix the brain
            optimizer.step()

        # QUIZ
        # example: the AI finishes studying for now we give it a quiz on new sentences
        valProbs, valLabels = GetPredictions(model, valLoader, device)
        
        # example: we test different lvls of how muuch the AI is sure 
        # does the AI need to be 10% sure or 90% sure to call a sentence toxic?
        # we pick the lvl that gives the best grade
        bestThEpoch, valF1 = 0.5, 0.0
        for th in np.linspace(0.1, 0.9, 9):
            preds = (valProbs >= th).astype(int)
            score = f1_score(valLabels, preds, zero_division=0)
            if score > valF1:
                valF1, bestThEpoch = score, th
        
        print(f"[Epoch {epoch}] quiz score: {valF1:.4f} (best lvl: {bestThEpoch:.2f})")

        # example: If this is the highest score weve ever seen save this brain
        if valF1 > bestValF1:
            bestValF1 = valF1
            torch.save(model.state_dict(), CFG.KCheckpointFile)

    # EXAM LAST
    # example: studying is over we load the smartest version of the brain we saved
    model.load_state_dict(torch.load(CFG.KCheckpointFile))
    
    # example: we test it one last time on sentences it has never seen before
    testProbs, testLabels = GetPredictions(model, testLoader, device)
    
    # example: use the lvl of how much AI is sure from the quizzes to make the final guesses
    testPreds = (testProbs >= bestThEpoch).astype(int)

    print("\n exam res")
    print(f"ttl correct: {accuracy_score(testLabels, testPreds):.4f}")
    print(f"toxic detection Score (F1): {f1_score(testLabels, testPreds, zero_division=0):.4f}")

    # example: save the finished brain into a folder so we can use it on a website or app later
    model.save_pretrained(CFG.KOutputDir)
    tokenizer.save_pretrained(CFG.KOutputDir)
    print("Finshied")

if __name__ == "__main__":
    Train()