import numpy

class rest_duration:
    DURATION=[0,1,2,4,8,16,32]
    """
    Possible rest durations:

    [0,1,2,4,8,16,32]
    """
    def quantize(durations):
        """Quantize each duration to its closest allowed rest duration."""
        return [rest_duration.DURATION[numpy.abs(numpy.subtract(rest_duration.DURATION, duration)).argmin()] for duration in durations]

class note_duration:
    DURATION=[0.25,0.5,0.75,1,1.5,2,3,4,6,8,16,32]
    """
    Possible note durations:

    [0,1,2,4,8,16,32]
    """
    def quantize(durations):
        """Quantize each duration to its closest allowed note duration."""
        return [note_duration.DURATION[numpy.abs(numpy.subtract(note_duration.DURATION, duration)).argmin()] for duration in durations]


def test():
    test_durations = [3,2,1.3,4,0.1,0.2,37]
    assert rest_duration.quantize(test_durations) == [2, 2, 1, 4, 0, 0, 32]
    assert note_duration.quantize(test_durations) == [3, 2, 1.5, 4, 0.25, 0.25, 32]
if __name__ == '__main__':
    test()