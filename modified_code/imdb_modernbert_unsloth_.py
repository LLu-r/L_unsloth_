import os

#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import unsloth
import torch
import sys
import logging
import evaluate

import pandas as pd
import numpy as np

from unsloth import FastModel, FastLanguageModel
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, training_args, \
    DataCollatorWithPadding
from datasets import Dataset

from sklearn.model_selection import train_test_split

train = pd.read_csv("/kaggle/input/labeledtraindata/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/testdata/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train, val = train_test_split(train, test_size=.2)

    train = train[0:20]
    val = val[0:20]
    test = test[0:20]

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    # model_name = 'answerdotai/ModernBERT-large'
    model_name = "microsoft/deberta-v2-xxlarge"
    NUM_CLASSES = 2

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=False,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        auto_model=AutoModelForSequenceClassification,
        num_labels=NUM_CLASSES,
        gpu_memory_utilization=0.8  # Reduce if out of memory
    )

    model = FastModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        # target_modules=[
        #     "q_proj", "k_proj", "v_proj", "o_proj",
        #     "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=32,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        task_type="SEQ_CLS",
    )

    print("model parameters:" + str(sum(p.numel() for p in model.parameters())))

    # make all parameters trainable
    # for param in model.parameters():
    #     param.requires_grad = True

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    def tokenize_function(examples):
        # return tokenizer(examples['text'])
        return tokenizer(examples['text'], max_length=512, truncation=True)


    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,

        warmup_steps=10,
        fp16=True,#
        bf16=False,#
        optim="adamw_8bit",#
        learning_rate=2e-5,
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        num_train_epochs=3,  # bert-style models usually need more than 1 epoch
        save_strategy="epoch",

        # report_to="wandb",

        # group_by_length=True,
        # eval_strategy="no",
        eval_strategy="no",
        # eval_steps=0.25,
        logging_strategy="steps",
        logging_steps=10,

        label_names=["label"],

        # 防止kaggle多进程死锁
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
    )

    from typing import Any, Dict, Tuple, Optional
    class UnslothSafeTrainer(Trainer):
        """
        安全的 prediction_step：**不** 调用 model.prediction_step（避免 unsloth 的 monkey-patch）。
        直接使用 model(**inputs) 并提取 loss/logits/labels。
        """

        def prediction_step(
                self,
                model,
                inputs: Dict[str, Any],
                prediction_loss_only: bool,
                ignore_keys: Optional[Tuple[str]] = None,
        ):
            # 把 inputs 放到设备上（复用 Trainer 的内部方法）
            inputs = self._prepare_inputs(inputs)

            # 获取是否有 labels（有时 key 叫 labels，有时叫 label）
            labels = None
            if isinstance(inputs, dict):
                labels = inputs.get("labels", inputs.get("label", None))

            # 直接用 forward（不要调用 model.prediction_step）
            with torch.no_grad():
                outputs = model(**inputs)

            # 解析 outputs：支持 dict 或 tuple
            loss = None
            logits = None

            # 如果 model 返回 dict（huggingface 习惯），尽量取标准字段
            if isinstance(outputs, dict):
                # loss 可能在 outputs['loss']
                loss = outputs.get("loss", None)
                # logits 可能在 outputs['logits'] 或 outputs['predictions']
                logits = outputs.get("logits", outputs.get("predictions", None))
            elif isinstance(outputs, tuple):
                # tuple 常见格式: (logits, ...) 或 (loss, logits, ...)
                if len(outputs) == 0:
                    logits = None
                elif len(outputs) == 1:
                    logits = outputs[0]
                elif len(outputs) == 2:
                    # 一般 (loss, logits) 或 (logits, labels) —— 尝试智能判断
                    a, b = outputs[0], outputs[1]
                    # 若 a 是标量 tensor 且 b 不是标量，猜 a 是 loss
                    if getattr(a, "ndim", None) == 0 or (isinstance(a, torch.Tensor) and a.numel() == 1):
                        loss = a
                        logits = b
                    else:
                        logits = a
                else:
                    # 常见 (loss, logits, ...) 或 (logits, ...)
                    # 优先把第一个标量当 loss，第二个当 logits
                    if getattr(outputs[0], "ndim", None) == 0 or (
                            isinstance(outputs[0], torch.Tensor) and outputs[0].numel() == 1):
                        loss = outputs[0]
                        logits = outputs[1] if len(outputs) > 1 else None
                    else:
                        logits = outputs[0]

            # 有时候 loss 在 outputs.loss，但未转到 cpu，保持 tensor
            if prediction_loss_only:
                # 只返回 loss 情况
                return (loss, None, None)

            # 确保 logits、labels 都在 cpu/device 上与 Trainer 期望一致
            return (loss, logits, labels)

    trainer = UnslothSafeTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer_stats = trainer.train()

    print(trainer_stats)

    # model.eval()
    # FastLanguageModel.for_inference(model)

    prediction_outputs = trainer.predict(test_dataset)
    print(prediction_outputs)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("deberta_unsloth.csv", index=False, quoting=3)
    logging.info('result saved!')