from enum import IntEnum

class Finger(IntEnum):
    Thumb = 0
    Index = 1
    Middle = 2
    Ring = 3
    Little = 4
    
    @staticmethod
    def get_array_of_points(finger):
        finger_array = None
        if finger == Finger.Thumb:
            finger_array = [(0, 4), (4, 3), (3, 2), (2, 1)]
        elif finger == Finger.Index:
            finger_array = [(0, 8), (8, 7), (7, 6), (6, 5)]
        elif finger == Finger.Middle:
            finger_array = [(0, 12), (12, 11), (11, 10), (10, 9)]
        elif finger == Finger.Ring:
            finger_array = [(0, 16), (16, 15), (15, 14), (14, 13)]
        else:
            finger_array = [(0, 20), (20, 19), (19, 18), (18, 17)]
        return finger_array
    
    @staticmethod
    def get_finger_name(finger):
        finger_name = ''
        if finger == Finger.Thumb:
            finger_name = 'Thumb'
        elif finger == Finger.Index:
            finger_name = 'Index'
        elif finger == Finger.Middle:
            finger_name = 'Middle'
        elif finger == Finger.Ring:
            finger_name = 'Ring'
        elif finger == Finger.Little:
            finger_name = 'Little'
        return finger_name