class AstrometricObservation:
    def __init__(self, t, rho, theta, sigma_rho, sigma_theta):
        self.t = t
        self.rho = rho
        self.theta = theta
        self.sigma_rho = sigma_rho
        self.sigma_theta = sigma_theta