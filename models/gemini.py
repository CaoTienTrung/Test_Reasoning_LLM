import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "Data"))
from custom_dataset import MultipleChoiceDataset

import time
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def append_to_json(file_path, result):
    with open(file_path, 'a', encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
        file.write('\n')

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None: 
        return False
    if "Response Error" in response:
        return False
    return True

class Gemini_Model:
    def __init__(self, keys: list, model_name="gemini-1.5-flash", patience=20, sleep_time=5):
        """
        Initializes the model with the API key and model name.

        Args:
            model (str): The name of the model to use.
            api_key (str): The API key for authentication.
            patience (int): The number of times to try
            sleep_time(int): Time to sleep when not response
        """
        self.keys = keys
        self.model_name = model_name
        self.patience = patience
        self.sleep_time = sleep_time

    def generate_response(self,  qas_dataset:MultipleChoiceDataset, file_path):
        key_id = 0
        GOOGLE_API_KEY = self.keys[key_id]
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(self.model_name)
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
            
            patience = self.patience               
            while not success:
                try:
                    response = model.generate_content(f'''
                        {input_data}
                        Hãy chọn đáp án đúng, ghi ngắn gọn A, B, C hay D, sau đó cho lời giải thích ngắn gọn.
                    ''')
                    print(id, response.text)

                    result = {
                        'id' : id,
                        'response' : response.text,
                        'correct answer' : qa.correct_answer
                    }

                    append_to_json(file_path, result )

                    success = True  
                except Exception as e:
                    print(f"Lỗi: {e}. Thử lại sau 5 giây.")
                    time.sleep(self.sleep_time)  

                    patience -=1
                    if patience <= 0:
                        key_id +=1
                        if key_id <= len(self.keys):
                            GOOGLE_API_KEY = self.keys[key_id]
                            genai.configure(api_key=GOOGLE_API_KEY)
                            model = genai.GenerativeModel(self.model_name)
                        else: 
                             raise RuntimeError("Out of API keys")

def main():
    API_keys = [
        "AIzaSyCuO6Es5f2WsdG7s6u1NqpevvJIKOFVHzY",
        'AIzaSyDPUvj6wbRbwxjI8_a2lqcf19qI2TOFAmk',
        'AIzaSyA3ltQitqfGSG9IYgssmnbXEMgpHn-uA3E',
        'AIzaSyAzpvsp5jnErWJb5vgfaT2Okq2aRfUrSao',
        'AIzaSyCy__vhtayjsOu7WpY1LDG3_13gvdZa_ds',
        'AIzaSyCuO6Es5f2WsdG7s6u1NqpevvJIKOFVHzY',
        'AIzaSyCpmEEbTWK13uooiTwDKWsHGIQ3gUMAvt8',
        'AIzaSyDlb2qrUqcaZpK2ZzvH7qhvEMVMS62kEQg',
        'AIzaSyCf2S8r16HLWWNGYZfNfCufaOsbKAaEmzU',
        'AIzaSyArdHQ-fXHBbM1zmatvgDCzNM5bQQEozTE',
        'AIzaSyC1oCt8EohMHozd7MjVdA-Q6x-PDIEb-fY',
        'AIzaSyAbs6OIp_D8vACldF-GFNZVjBuZQTqfguU'
    ]

    model = Gemini_Model(keys=API_keys, 
                         model_name="gemini-1.5-flash", 
                         patience=10, 
                         sleep_time=5
                        )

    QAS_JSON_FILE_PATH = r"F:/LLMs/Data/qas.json" 
    SGK_JSON_FILE_PATH = r"F:/LLMs/Data/sgk.json"
    subject = 'D'
    dataset = MultipleChoiceDataset(QAS_JSON_FILE_PATH, SGK_JSON_FILE_PATH)
    # sub_dataset = dataset.__getBaseOnSubject__(subject)
    new_dataset = dataset.__getQuestionFromIdToEnd__('G12B0703260')

    SAVE_PATH = r"F:\LLMs\Resut\gemini_response.json"
    model.generate_response(new_dataset, SAVE_PATH)

if __name__ == "__main__":
    main()