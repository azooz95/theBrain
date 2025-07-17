

class SeqPipline():
    '''
        inputs from one step goes to another step
    '''
    def __init__(self):
        self.steps = []
        self.extra_steps = []


    def add_step(self, func , *args, **kwargs):
        self.steps.append((func, args, kwargs))

    def clean_results(self):
        self.output = []

    def run(self, input_data):
        self.output = []
        """Run the pipeline by passing data through each step."""
        for func, args, kwargs in self.steps:
           
            is_return = kwargs.get('is_return', False)
            
            if is_return: 
                del kwargs['is_return']

            input_data = func(input_data, *args, **kwargs)

            if is_return: 
                self.output.append(input_data)

        return self.output