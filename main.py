from training import training
from dataset import get_data
from model import model_Quantization
#from evaluate import evaluate
def run(train_loader,model):
    trainer=training(train_loader,model)
    trainer.train()
    



if __name__=="__main__":
    model,tokenizer=model_Quantization()
    train_data, test_data, train_loader, test_loader=get_data('EnglishDataset/data.csv',tokenizer)
    run(train_loader,model)



   