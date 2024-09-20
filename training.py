from transformers import TrainingArguments,Trainer

def Training_Arguments():
    args = TrainingArguments(
        output_dir="bloomspellCorrection",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=True,
        optim="adafactor",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        report_to="tensorboard",  
    )
    return args

def training(train_loader,model):
    args = Training_Arguments()
    
    trainer = Trainer(
        args=args,
        train_dataset=train_loader.dataset,
        model=model,
    )
    
    return trainer
