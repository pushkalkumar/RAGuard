class Defense:
    def __init__(self):
        # creating an empty placeholder constructor
        pass
    def detect_poison(self,data):
        # currently a placeholder for actual detection logic
        return [line.strip() for line in data if "malicious" in line]  # dummy logic