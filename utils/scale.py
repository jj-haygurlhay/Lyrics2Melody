import numpy

class scale:
    """
    Represents a scale in the sense of a note appartenance class.
    """
    MAJOR_SCALE=[0,2,4,5,7,9,11]
    NATMIN_SCALE=[0, 2, 3, 5, 7, 8, 10]
    NATMIN_SCALE=[0,2,3,5,7,8,11]
    def __init__(self, note_pattern: list[int], offset: int=0):
        """
        Creates a scale.

        Parameters
        ----------
        note_pattern : a scale's note appartenance class. Can be used as a "shape" (major and minor) for the scale when used with offset.

        offset : used to offset a specified "shape".
        """
        self.note_pattern = [i%12 for i in note_pattern]
        self.offset = offset
    
    def __contains__(self, note: int|float):
        return (note%12 + self.offset) in self.note_pattern
    
    def __iter__(self):
        for i in range(0, 128):
            if(i in self):
                yield i
    
    def difference(self, note: int|float):
        """
        Returns the magnitude and direction (+/-) to the closest note in the scale.
        """
        note = (note-self.offset)%12
        note_diff = numpy.subtract(self.note_pattern, note)
        return note_diff[numpy.argmin(numpy.abs(note_diff))]

    def melody_appartenance(self, notes: list[int]|list[float], durations:list[float]):
        """
        Gives a score that represent how much a given melody is off this scale.

        0 is the best score.
        """
        if len(notes) != len(durations): raise Exception("Cannot implicitly create a melody from missmatched number of notes and duration values.")
        error_sum = 0
        duration_sum = 0
        for i in range(len(notes)):
            error_sum += abs(self.difference(notes[i]))*durations[i]
            duration_sum += durations[i]
        return error_sum/duration_sum
    
    def fit_to_scale(self, notes: list[int]|list[float]):
        """
        Takes a sequence of notes and fits it to this scale.
        """
        return [round(note + self.difference(note)) for note in notes]
    
def find_closest_fit(notes, durations, scales):
    appartenances = [scale_candidate.melody_appartenance(notes, durations) for scale_candidate in scales]
    return scales[numpy.argmin(appartenances)], min(appartenances)

ALL_SCALES = [scale(scale.MAJOR_SCALE, i) for i in range(12)] + [scale(scale.NATMIN_SCALE, i) for i in range(12)] + [scale(scale.NATMIN_SCALE, i) for i in range(12)]

def test():
    test_scale = scale(scale.MAJOR_SCALE, 0)
    test_notes = [0,1,2,3,3.5,4,6,13.2]
    assert 14 in test_scale
    assert 14.0 in test_scale
    assert not 13 in test_scale
    assert not 13.0 in test_scale
    assert test_scale.difference(13.2) - float(0.8) < 0.00001 #Floating point error
    assert test_scale.difference(12.8) + float(0.8) < 0.00001 #Floating point error
    assert test_scale.fit_to_scale(test_notes) == [0, 0, 2, 2, 4, 4, 5, 14]
    # test_notes = [i +10 for i in scale.MAJOR_SCALE]
    # closest_scale = find_closest_fit(test_notes, [1]*7, ALL_SCALES)[0]
    # print([closest_scale.difference(note) for note in test_notes])



if __name__ == '__main__':
    test()