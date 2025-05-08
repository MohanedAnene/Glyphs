class GlyphAgent:
    def __init__(self, glyph_filename:str, model_filename:str, name:str=None):
        if name is None:
            self.name = glyph_filename
        else:
            self.name = name

        return
        # works on a glyph file (.mglyph/.zip)
        # and a NN model loaded from a file

        # the construtor should load the model and check the existence of the glyph file

    def get_response(self, task:dict) -> dict:
        # returns a decision based on the task
        # task is a dictionary with the following keys:
        # {'x1': x1, 'x2': x2, 'distance': self.current_distance}

        # this function should use the model to predict the values of the two glyphs loaded from the glyph file
        # and return the decision based on the task
        # the glyphs sizes x1, x2 given in the task are the sizes of the glyphs in the glyph file
        # they are float numbers, so the function shoudl find the nearest available glyph sizes
        # these will be given to the model to predict the values of the two glyphs
        return {'choice': random.choice(['<', '>', '==']), 'time': '-/', 'glyph-name': self.name}
    