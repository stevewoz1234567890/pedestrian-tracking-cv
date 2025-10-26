"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import numpy as np

from ps5_utils import run_kalman_filter, run_particle_filter

# I/O directories
input_dir = "input"
output_dir = "output"



# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state [x, y, vx, vy]
        
        # State transition matrix (4x4)
        self.D = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=float)
        
        # Measurement matrix (2x4) - maps state to measurements
        self.M = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=float)
        
        # Process noise covariance
        self.Q = Q
        
        # Measurement noise covariance
        self.R = R
        
        # State covariance matrix
        self.P = np.eye(4) * 100  # Initial uncertainty

    def predict(self):
        """Predict the next state and covariance"""
        # State prediction: x_t = D * x_{t-1}
        self.state = self.D @ self.state
        
        # Covariance prediction: P_t = D * P_{t-1} * D^T + Q
        self.P = self.D @ self.P @ self.D.T + self.Q

    def correct(self, meas_x, meas_y):
        """Correct the state using measurements"""
        # Measurement vector
        z = np.array([meas_x, meas_y])
        
        # Predicted measurement
        z_pred = self.M @ self.state
        
        # Innovation (residual)
        y = z - z_pred
        
        # Innovation covariance
        S = self.M @ self.P @ self.M.T + self.R
        
        # Kalman gain
        K = self.P @ self.M.T @ np.linalg.inv(S)
        
        # State correction
        self.state = self.state + K @ y
        
        # Covariance correction
        I = np.eye(4)
        self.P = (I - K @ self.M) @ self.P

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        
        # Initialize particles uniformly around the template location
        template_h, template_w = self.template.shape[:2]
        center_x = self.template_rect['x'] + self.template_rect['w'] // 2
        center_y = self.template_rect['y'] + self.template_rect['h'] // 2
        
        # Initialize particles with some spread around the template center
        self.particles = np.zeros((self.num_particles, 2))
        self.particles[:, 0] = np.random.normal(center_x, 20, self.num_particles)  # x coordinates
        self.particles[:, 1] = np.random.normal(center_y, 20, self.num_particles)  # y coordinates
        
        # Initialize weights uniformly
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Store template dimensions for cutout
        self.template_h = template_h
        self.template_w = template_w

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        # Calculate Mean Squared Error
        mse = np.mean((template.astype(float) - frame_cutout.astype(float)) ** 2)
        
        # Convert MSE to similarity using exponential (higher similarity = lower MSE)
        similarity = np.exp(-mse / (2 * self.sigma_exp ** 2))
        
        return similarity

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        # Normalize weights to ensure they sum to 1
        weights_normalized = self.weights / np.sum(self.weights)
        
        # Resample particles based on weights
        indices = np.random.choice(self.num_particles, size=self.num_particles, 
                                 p=weights_normalized, replace=True)
        
        return self.particles[indices]

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        # Add dynamics noise to particles
        noise_x = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_y = np.random.normal(0, self.sigma_dyn, self.num_particles)
        self.particles[:, 0] += noise_x
        self.particles[:, 1] += noise_y
        
        # Keep particles within image bounds
        h, w = frame.shape[:2]
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)
        
        # Update weights based on similarity
        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])
            
            # Check if particle is within bounds for template extraction
            if (x - self.template_w//2 >= 0 and x + self.template_w//2 < w and
                y - self.template_h//2 >= 0 and y + self.template_h//2 < h):
                
                # Extract patch from frame
                patch = frame[y - self.template_h//2:y + self.template_h//2,
                            x - self.template_w//2:x + self.template_w//2]
                
                # Resize patch to match template size if needed
                if patch.shape != self.template.shape:
                    patch = cv2.resize(patch, (self.template_w, self.template_h))
                
                # Calculate similarity
                similarity = self.get_error_metric(self.template, patch)
                self.weights[i] = similarity
            else:
                # Particle is out of bounds, set weight to 0
                self.weights[i] = 0
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            # If all weights are 0, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Resample particles
        self.particles = self.resample_particles()

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Draw particles as colored dots
        for i in range(self.num_particles):
            x, y = int(self.particles[i, 0]), int(self.particles[i, 1])
            cv2.circle(frame_in, (x, y), 2, (0, 255, 0), -1)
        
        # For MDParticleFilter, use weighted mean size for rectangle
        if hasattr(self, 'particles') and self.particles.shape[1] == 4:
            # Multi-dimensional particles: use weighted mean size
            w_weighted_mean = np.sum(self.particles[:, 2] * self.weights)
            h_weighted_mean = np.sum(self.particles[:, 3] * self.weights)
            cv2.rectangle(frame_in, 
                          (int(x_weighted_mean - w_weighted_mean//2), 
                           int(y_weighted_mean - h_weighted_mean//2)),
                          (int(x_weighted_mean + w_weighted_mean//2), 
                           int(y_weighted_mean + h_weighted_mean//2)),
                          (255, 0, 0), 2)
        else:
            # Regular particles: use template size
            cv2.rectangle(frame_in, 
                          (int(x_weighted_mean - self.template_w//2), 
                           int(y_weighted_mean - self.template_h//2)),
                          (int(x_weighted_mean + self.template_w//2), 
                           int(y_weighted_mean + self.template_h//2)),
                          (255, 0, 0), 2)
        
        # Calculate and draw spread circle
        distances = np.sqrt((self.particles[:, 0] - x_weighted_mean)**2 + 
                          (self.particles[:, 1] - y_weighted_mean)**2)
        weighted_std = np.sum(distances * self.weights)
        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), 
                  int(weighted_std), (0, 0, 255), 2)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        # First run the standard particle filter process
        super().process(frame)
        
        # Update the appearance model using IIR filter
        # Calculate weighted mean position
        x_weighted_mean = np.sum(self.particles[:, 0] * self.weights)
        y_weighted_mean = np.sum(self.particles[:, 1] * self.weights)
        
        # Extract the best patch at the weighted mean position
        x, y = int(x_weighted_mean), int(y_weighted_mean)
        h, w = frame.shape[:2]
        
        if (x - self.template_w//2 >= 0 and x + self.template_w//2 < w and
            y - self.template_h//2 >= 0 and y + self.template_h//2 < h):
            
            best_patch = frame[y - self.template_h//2:y + self.template_h//2,
                             x - self.template_w//2:x + self.template_w//2]
            
            # Resize patch to match template size if needed
            if best_patch.shape != self.template.shape:
                best_patch = cv2.resize(best_patch, (self.template_w, self.template_h))
            
            # Update template using IIR filter: template = (1-alpha) * template + alpha * best_patch
            self.template = (1 - self.alpha) * self.template + self.alpha * best_patch


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        
        # For multi-dimensional particle filter, particles should track (x, y, w, h)
        # Initialize particles with size information
        template_h, template_w = self.template.shape[:2]
        center_x = self.template_rect['x'] + self.template_rect['w'] // 2
        center_y = self.template_rect['y'] + self.template_rect['h'] // 2
        
        # Initialize particles with position and size
        self.particles = np.zeros((self.num_particles, 4))  # [x, y, w, h]
        self.particles[:, 0] = np.random.normal(center_x, 20, self.num_particles)  # x
        self.particles[:, 1] = np.random.normal(center_y, 20, self.num_particles)  # y
        self.particles[:, 2] = np.random.normal(template_w, 5, self.num_particles)  # width
        self.particles[:, 3] = np.random.normal(template_h, 5, self.num_particles)  # height
        
        # Ensure positive sizes
        self.particles[:, 2] = np.maximum(self.particles[:, 2], 10)
        self.particles[:, 3] = np.maximum(self.particles[:, 3], 10)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        # Add dynamics noise to particles (position and size)
        noise_x = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_y = np.random.normal(0, self.sigma_dyn, self.num_particles)
        noise_w = np.random.normal(0, self.sigma_dyn/2, self.num_particles)  # smaller noise for size
        noise_h = np.random.normal(0, self.sigma_dyn/2, self.num_particles)
        
        self.particles[:, 0] += noise_x
        self.particles[:, 1] += noise_y
        self.particles[:, 2] += noise_w
        self.particles[:, 3] += noise_h
        
        # Keep particles within image bounds
        h, w = frame.shape[:2]
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, h - 1)
        
        # Ensure positive sizes and reasonable bounds
        self.particles[:, 2] = np.clip(self.particles[:, 2], 10, w//2)
        self.particles[:, 3] = np.clip(self.particles[:, 3], 10, h//2)
        
        # Update weights based on similarity
        for i in range(self.num_particles):
            x, y, w_p, h_p = int(self.particles[i, 0]), int(self.particles[i, 1]), int(self.particles[i, 2]), int(self.particles[i, 3])
            
            # Check if particle is within bounds for template extraction
            if (x - w_p//2 >= 0 and x + w_p//2 < w and
                y - h_p//2 >= 0 and y + h_p//2 < h):
                
                # Extract patch from frame
                patch = frame[y - h_p//2:y + h_p//2, x - w_p//2:x + w_p//2]
                
                # Resize patch to match template size
                if patch.shape != self.template.shape:
                    patch = cv2.resize(patch, (self.template_w, self.template_h))
                
                # Calculate similarity
                similarity = self.get_error_metric(self.template, patch)
                self.weights[i] = similarity
            else:
                # Particle is out of bounds, set weight to 0
                self.weights[i] = 0
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)
        else:
            # If all weights are 0, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Resample particles
        self.particles = self.resample_particles()
        
        # Update appearance model using IIR filter
        # Calculate weighted mean position and size
        x_weighted_mean = np.sum(self.particles[:, 0] * self.weights)
        y_weighted_mean = np.sum(self.particles[:, 1] * self.weights)
        w_weighted_mean = np.sum(self.particles[:, 2] * self.weights)
        h_weighted_mean = np.sum(self.particles[:, 3] * self.weights)
        
        # Extract the best patch at the weighted mean position and size
        x, y, w_p, h_p = int(x_weighted_mean), int(y_weighted_mean), int(w_weighted_mean), int(h_weighted_mean)
        
        if (x - w_p//2 >= 0 and x + w_p//2 < w and
            y - h_p//2 >= 0 and y + h_p//2 < h):
            
            best_patch = frame[y - h_p//2:y + h_p//2, x - w_p//2:x + w_p//2]
            
            # Resize patch to match template size if needed
            if best_patch.shape != self.template.shape:
                best_patch = cv2.resize(best_patch, (self.template_w, self.template_h))
            
            # Update template using IIR filter
            self.template = (1 - self.alpha) * self.template + self.alpha * best_patch


def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 100  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 150  # Define the number of particles
    sigma_mse = 15  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 8  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 200  # Define the number of particles
    sigma_mse = 12  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 6  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.1  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 300  # Define the number of particles
    sigma_md = 20  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.1  # Set a value for alpha

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out
