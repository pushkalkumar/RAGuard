from retrievers.retriever import Retriever
from defences.defense import Defense

def main():
    #Load the clean data
    retriever_clean_data = Retriever('datasets/clean/sample_clean.txt')
    retriever_clean_data.load_data()

    #Load the poisoned data
    retriever_poisoned_data = Retriever('datasets/poisoned/sample_poisoned.txt')
    retriever_poisoned_data.load_data()

    #Run the defense mechanism
    defense = Defense()
    poisoned_lines = defense.detect_poison(retriever_poisoned_data.data)

    print("Detected Poisoned Files:")
    for line in poisoned_lines:
        print("-", line)


if __name__ == "__main__":
    main()