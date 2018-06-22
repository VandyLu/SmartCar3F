import numpy as np

class PID():
    ''' A naive PID implementation '''

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.int_e = 0.0
        self.last_e = 0.0

    def update(self, err):
        output = self.kp * err + self.ki * self.int_e + self.kd * (err - self.last_e)
        # Update integral and derivative
        self.int_e += err
        self.last_e = err

        self.int_e = np.clip(self.int_e, -30, 30)
        return output
