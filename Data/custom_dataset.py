import json
import torch
from torch.utils.data import DataLoader, Dataset

class Question: 
    def __init__(self, context: str, question: str, options: list, correct_answer: str, explanation: str):
        '''
                """
        Initializes Question.

        Args:
            context (str): The context that system need to rely on to answer question.
            question (str): question need to answer
            options (list): list of four options
            correct_answer (str): the correct option
            explanation (str): the reason to choose the correct answer
        """
        '''
        self.context = context
        self.question = question
        self.options = options
        self.correct_answer = correct_answer
        self.explanation = explanation

class MultipleChoiceDataset(Dataset):
    def __init__(self, qas_json_file=None, sgk_json_file=None):
        self.questions = dict()
        self.sgk = dict()

        if qas_json_file is not None and sgk_json_file is not None : 
            self.loadDatasetFromFile(qas_json_file, sgk_json_file)

    def loadDatasetFromFile(self, qas_json_file, sgk_json_file):
        with open(sgk_json_file, "r", encoding="utf-8") as file_sgk:
            sgk_data = json.load(file_sgk)
            self._loadSgk(sgk_data)

        with open(qas_json_file, "r", encoding="utf-8") as file_qas:
            qas_data = json.load(file_qas)
            self._loadQas(qas_data)

    def _loadSgk(self, sgk_data):
        for subject, grade_items in sgk_data.items():
           for grade, chapter_items in  grade_items.items():
               for chapter, lesson_items in chapter_items.items():
                   for lesson in lesson_items.keys():
                        if lesson != "name":
                            self.sgk[f"{subject}{grade}{lesson}"] = lesson_items[lesson]["context"]

    def _loadQas(self, qas_data):
        for id, question_data in qas_data.items():
            question = question_data["question"]
            options = question_data["answer_options"]
            correct_answer = question_data["correct_answer"]
            explanation = question_data["explanation"]
            
            try: 
                context = self.sgk[id[:6]]
            except:
                context = ""


            self.questions[id] = Question(context, question, options, correct_answer, explanation)

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        return self.questions[index]
    
    def __getBaseOnSubject__(self, subject):
        sub_ques = {id: question for id, question in self.questions.items() if id[0] == subject}
        new_dataset = MultipleChoiceDataset()
        new_dataset.questions = sub_ques
        return new_dataset
    
    def __getQuestionFromIdToEnd__(self, curr_id):
        unprompt_ques = {id: question for id, question in self.questions.items() if id[:3] == curr_id[:3] and id[3:] >= curr_id[3:]}
        new_dataset = MultipleChoiceDataset()
        new_dataset.questions = unprompt_ques
        return new_dataset