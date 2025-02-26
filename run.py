import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'Data'))

from custom_dataset import MultipleChoiceDataset

def main():
    QAS_JSON_FILE_PATH = "F:/LLMs/Data/qas.json" 
    SGK_JSON_FILE_PATH = "F:/LLMs/Data/sgk.json"

    data = MultipleChoiceDataset(QAS_JSON_FILE_PATH, SGK_JSON_FILE_PATH)
    print(data)

if __name__ == "__main__":
    main()