# new class
# For Loss: 
#    - plots the loss
#    - logs the loss for training and validation 
# For Model:
#    - saves the model

class log():
    def __init__(self, args):
        # create the log file 
        self.args = args