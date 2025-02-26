import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "Data"))
from custom_dataset import MultipleChoiceDataset

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def append_to_json(file_path, result):
    with open(file_path, 'a', encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
        file.write(',','\n')

class Vina_Llama_Model:
    def __init__(self, model_path="vilm/vinallama-7b-chat", device="cuda", max_new_tokens=100):
        """
        Initializes the model with model path.

        Args:
            model_path (str): The path of the model to use.
            device (str): The device to load model
            max_new_tokens (int): Max tokens of model generation
        """

        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

    def generate_response(self,  qas_dataset:MultipleChoiceDataset):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()

        for id, qa in qas_dataset.questions.items():
            success = False

            input_data = {
                'question': qa.question,
                'options': {
                    'A': qa.options[0],
                    'B': qa.options[1],
                    'C': qa.options[2],
                    'D': qa.options[3],
                }
            }

            prompts = [f"system\nBạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác bằng cách chọn đáp án đúng, ghi ngắn gọn A, B, C hay D, sau đó cho lời giải thích ngắn gọn. .\nuser\n{input_data}\nassistant\n" for input_text in user_inputs]
        
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Sinh văn bản từ mô hình
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens
            )
            
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # responses = [response.split("assistant\n")[1].strip() if "assistant\n" in response else response for response in responses]
            print(responses)
            return responses
def main():
    model = Vina_Llama_Model()

    QAS_JSON_FILE_PATH = r"F:/Project/LLMs/Data/qas.json" 
    SGK_JSON_FILE_PATH = r"F:/Project/LLMs/Data/sgk.json"
    # subject = 'D'
    dataset = MultipleChoiceDataset(QAS_JSON_FILE_PATH, SGK_JSON_FILE_PATH)
    # sub_dataset = dataset.__getBaseOnSubject__(subject)
    # new_dataset = dataset.__getQuestionFromIdToEnd__('G12B0703260')

    # SAVE_PATH = r"F:\LLMs\Resut\gemini_response.json"
    model.generate_response(dataset)

if __name__ == "__main__":
    main()