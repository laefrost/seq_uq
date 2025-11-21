from datasets import Dataset

class TextDataset(Dataset):
    def __init__(self, arrow_table, info = None, split = None, indices_table = None, fingerprint = None):
        super().__init__(arrow_table, info, split, indices_table, fingerprint)
        
        