class Retriever:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = []
    def load_data(self):
        with open (self.dataset_path, 'r') as file:
            self.data = file.readlines()
    def query(self,text):
        # currently a placeholder for actual retrieval logic
        return self.data[:5]  # returns first 5 lines as dummy data