class PID():

    """ Called from the children of PID_Framework"""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        # Initially, the "planned starting of trajectory" coincides with the "actual starting position".
        self.accumulated_error = 0
        self.last_error = 0

    def increment_intregral_error(self, error, pi_limit=3):
        self.accumulated_error = self.accumulated_error + error
        if (self.accumulated_error > pi_limit):
            self.accumulated_error = pi_limit
        elif (self.accumulated_error < pi_limit):
            self.accumulated_error = -pi_limit

    def compute_output(self, error, dt_error):
        self.increment_intregral_error(error)
        self.last_error = error
        return self.Kp * error + self.Ki * self.accumulated_error + self.Kd * dt_error