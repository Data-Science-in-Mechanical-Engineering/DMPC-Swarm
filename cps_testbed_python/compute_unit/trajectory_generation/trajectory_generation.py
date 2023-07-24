"""contains code for data generation

includes the computing node agent
"""
import math
from dataclasses import dataclass
from compute_unit.trajectory_generation.interpolation import TrajectoryCoefficients
from compute_unit.trajectory_generation.interpolation import Interpolation
import numpy as np
import qpsolvers as qp
from cvxopt import matrix, solvers
import copy
import time
import scipy.optimize as opt


@dataclass
class TrajectoryGeneratorOptions:
    """this class represents the options for the data generator

    Parameters
    ---------
        order_polynom: int
            order of interpolation polynom
        objective_function_sample_points: np.array
            objective function is integrated. However for this data generator it is approximated by summation
            over objective_function_sample_points.
        collision_constraint_sample_points: np.array
            collision constraint is evaluated at this points
        state_constraint_sample_points: np.array
            position/speed/acc/jerk constraint is evaluated at this points
        prediction_horizon: int
            how many time intervals to calculate (i.e. data will be delta_t_statespace*prediction_horizon long)
        weight_state_difference: np.array, shape(3, 3)
            positive definite weight matrix for difference between data and target state
        weight_state_derivative_1_difference: np.array, shape(3, 3)
            positive definite weight matrix for first order derivative (speed)
        weight_state_derivative_2_difference: np.array, shape(3, 3)
            positive definite weight matrix for second order derivative (acceleration)
        weight_state_derivative_3_difference: np.array, shape(3, 3)
            positive definite weight matrix for second order derivative (jerk)
        objective_function_sampling_delta_t: float
            time step for sampling the integration should be divide with delta_t_statespace by an integer
        max_position: np.array, shape(3,)
            maximum position the drone can reach
        max_speed: np.array, shape(3, )
            maximum speed the drone can reach
        max_acceleration: np.array, shape(3, )
            maximum accelecration the drone can reach
        max_jerk: np.array, shape(3, )
            maximum jerk the drone can reach
        min_position: np.array, shape(3,)
            minimum position the drone can reach
        r_min: float
            minimum distance between planned data and other trajectories
        optimization_variable_sample_time: float
            sample time of the optimization variable, Communication sample time has to be a integer multiple
        min_distance_cooperative: float
            minimum distance difference for coperative behavior

    """
    objective_function_sample_points: any
    objective_function_sample_points_pos: any
    collision_constraint_sample_points: any
    state_constraint_sample_points: any
    weight_state_difference: any
    weight_state_derivative_1_difference: any
    weight_state_derivative_2_difference: any
    weight_state_derivative_3_difference: any
    max_position: any
    max_speed: any
    max_acceleration: any
    max_jerk: any
    min_position: any
    r_min: float
    optimization_variable_sample_time: float
    num_drones: int
    skewed_plane_BVC: bool
    use_qpsolvers: bool
    downwash_scaling_factor: float
    use_soft_constraints: bool
    guarantee_anti_collision: bool
    soft_constraint_max: float
    weight_soft_constraint: float
    min_distance_cooperative: float
    weight_cooperative: float
    cooperative_normal_vector_noise: float


class TrajectoryGenerator:
    """generator for the trajectories
    Builds the optimization problem and solves it

    """

    def __init__(self, options,
                 trajectory_interpolation: Interpolation):
        """constructor
        Parameters
        ----------
            options: TrajectoryGeneratorOptions
                options fot the generator
            trajectory_interpolation:
                interpolation for data
        """
        self.__options = options
        self.__trajectory_interpolation = trajectory_interpolation
        self.__breakpoints = trajectory_interpolation.breakpoints
        self.__objective_function_sample_points = options.objective_function_sample_points
        self.__objective_function_sample_points_pos = options.objective_function_sample_points_pos
        self.__num_objective_function_sample_points = len(self.__objective_function_sample_points)
        self.__num_objective_function_sample_points_pos = len(self.__objective_function_sample_points_pos)
        self.__collision_constraint_sample_points = options.collision_constraint_sample_points
        self.__num_collision_constraint_sample_points = len(self.__collision_constraint_sample_points)
        self.__state_constraint_sample_points = options.state_constraint_sample_points
        self.__num_state_constraint_sample_points = len(self.__state_constraint_sample_points)
        self.__dim = self.__trajectory_interpolation.dimension

        self.__use_soft_constraints = int(options.use_soft_constraints)
        self.__soft_constraint_max = options.soft_constraint_max
        self.__guarantee_anti_collision = int(options.guarantee_anti_collision)

        self.__prediction_horizon = trajectory_interpolation.num_intervals

        self.__upper_boundaries = [options.max_position, options.max_speed, options.max_acceleration, options.max_jerk]
        self.__lower_boundaries = [options.min_position, -options.max_speed, -options.max_acceleration,
                                   -options.max_jerk]

        self.__num_anti_collision_constraints = (options.num_drones - 1) * self.__num_collision_constraint_sample_points
        self.__num_optimization_variables = trajectory_interpolation.num_optimization_variables + \
            self.__use_soft_constraints*self.__num_anti_collision_constraints * 0
        # initialize all matrixes with zeros (allocate their memory)
        # objective function = x' Q x + 2x' P
        self.__Q = np.zeros((self.__num_optimization_variables, self.__num_optimization_variables))

        self.__P = np.zeros((self.__dim * len(self.__objective_function_sample_points_pos),
                             self.__num_optimization_variables))

        # initialize Aeq, end constraints: speed, acceleration and jerk = 0 for feasibility
        self.__A_eq = np.zeros((trajectory_interpolation.num_equality_constraints + self.__dim * 3,
                                self.__num_optimization_variables))

        self.__b_eq = np.zeros((len(self.__A_eq),))

        # build inequality constraint
        self.__num_anti_collision_constraints = options.num_drones * self.__num_collision_constraint_sample_points
        num_unequality_constraints = 3 * self.__dim * 2 * self.__num_state_constraint_sample_points + \
                                     2 * self.__trajectory_interpolation.num_optimization_variables  # 2 for position, speed, acc TODO: jerk

        self.__A_uneq_part = np.zeros((num_unequality_constraints,
                                       self.__trajectory_interpolation.num_optimization_variables))
        self.__b_uneq_part = np.ones((num_unequality_constraints,))

        self.__coefficients_to_trajectory_matrix_objective = None
        self.__coefficients_to_trajectory_derivative_1_matrix_objective = None
        self.__coefficients_to_trajectory_derivative_2_matrix_objective = None
        self.__coefficients_to_trajectory_derivative_3_matrix = None

        self.__downwash_scaling = np.diag([1, 1, 1.0 / float(self.__options.downwash_scaling_factor)])

        self.__downwash_scaling_stacked = np.zeros((3*self.__num_collision_constraint_sample_points,
                                                    3*self.__num_collision_constraint_sample_points))
        for i in range(self.__num_collision_constraint_sample_points):
            self.__downwash_scaling_stacked[i*3:(i+1)*3, i*3:(i+1)*3] = self.__downwash_scaling

        self.__state_trajectory_vector_matrix_dt = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            [self.__trajectory_interpolation.breakpoints[-1]], derivative_order=1)

        self.__state_trajectory_vector_matrix_dtdt = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            [self.__trajectory_interpolation.breakpoints[-1]], derivative_order=2)

        self.__last_num_constraints = None

        self.__normal_vector_noise = options.cooperative_normal_vector_noise

        self.initialize_optimization()

    def initialize_optimization(self):
        """initializes optimization by building matrixes that are constant for every optimization instance"""

        # build matrix that allow to calculate position/speed at objective sample points out of the coefficients
        self.__coefficients_to_trajectory_matrix_objective = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__objective_function_sample_points_pos, derivative_order=0)
        self.__coefficients_to_trajectory_derivative_1_matrix_objective = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__objective_function_sample_points, derivative_order=1)
        self.__coefficients_to_trajectory_derivative_2_matrix_objective = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__objective_function_sample_points, derivative_order=2)

        # build matrix that allow to calculate position/speed at state constraints sample points out of the coefficients
        self.__coefficients_to_trajectory_matrix_state = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=0)
        self.__coefficients_to_trajectory_derivative_1_matrix_state = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=1)
        self.__coefficients_to_trajectory_derivative_2_matrix_state = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=2)

        # build matrix that allow to calculate position/speed at collision constraints sample points out of the
        # coefficients
        self.__coefficients_to_trajectory_matrix_collision = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
            self.__collision_constraint_sample_points, derivative_order=0)

        # build matrix that allow to calculate position/speed at objective sample points out of the initial state
        self.__state_to_trajectory_matrix_state = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=0)
        self.__state_to_trajectory_derivative_1_matrix_state = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=1)
        self.__state_to_trajectory_derivative_2_matrix_state = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__state_constraint_sample_points, derivative_order=2)

        # build matrix that allow to calculate position/speed at objective sample points out of the initial state
        self.__state_to_trajectory_matrix_objective = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__objective_function_sample_points_pos, derivative_order=0)

        # build matrix that allow to calculate position/speed at objective sample points out of the initial state
        self.__state_to_trajectory_matrix_collision = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(
            self.__collision_constraint_sample_points, derivative_order=0)

        # build objective function matrizes
        Q = np.zeros((self.__trajectory_interpolation.num_optimization_variables,
                      self.__trajectory_interpolation.num_optimization_variables))
        Q_tilde = np.kron(np.eye(self.__num_objective_function_sample_points_pos, dtype=int),
                          self.__options.weight_state_difference)
        Q += self.__coefficients_to_trajectory_matrix_objective.T @ Q_tilde @ \
                    self.__coefficients_to_trajectory_matrix_objective

        Q_tilde = np.kron(np.eye(self.__num_objective_function_sample_points, dtype=int),
                          self.__options.weight_state_derivative_1_difference)
        Q += self.__coefficients_to_trajectory_derivative_1_matrix_objective.T @ Q_tilde @ \
                    self.__coefficients_to_trajectory_derivative_1_matrix_objective

        Q_tilde = np.kron(np.eye(self.__num_objective_function_sample_points, dtype=int),
                          self.__options.weight_state_derivative_2_difference)
        Q += self.__coefficients_to_trajectory_derivative_2_matrix_objective.T @ Q_tilde @ \
                    self.__coefficients_to_trajectory_derivative_2_matrix_objective

        Q_tilde = np.kron(np.eye(self.__prediction_horizon), self.__options.weight_state_derivative_3_difference)
        Q += Q_tilde

        self.__Q[0:self.__trajectory_interpolation.num_optimization_variables,
                 0:self.__trajectory_interpolation.num_optimization_variables] = Q

        if self.__use_soft_constraints == 1:
            self.__Q[self.__trajectory_interpolation.num_optimization_variables:,
                     self.__trajectory_interpolation.num_optimization_variables:] = \
                np.kron(np.eye(self.__num_optimization_variables-self.__trajectory_interpolation.num_optimization_variables),
                        self.__options.weight_soft_constraint)

        P_tilde = np.kron(np.eye(self.__num_objective_function_sample_points_pos, dtype=int),
                          self.__options.weight_state_difference)
        self.__P[:, 0:self.__trajectory_interpolation.num_optimization_variables] = P_tilde @ \
            self.__coefficients_to_trajectory_matrix_objective


        # build equality matrizes
        # first from data, second start/end constraints
        a = self.__trajectory_interpolation.num_equality_constraints  # just to make the following shorter
        self.__A_eq[0:a, 0:self.__trajectory_interpolation.num_optimization_variables], self.__b_eq[0:a] = \
            self.__trajectory_interpolation.equality_constraint

        """
        for i in range(0, 3):
            self.__A_eq[a + self.__dim * i:a + self.__dim * (i + 1)] = \
                self.__trajectory_interpolation.get_trajectory_vector_matrix([0], derivative_order=i)
        """
        for i in range(0, 2):
            self.__A_eq[a + self.__dim * i: a + self.__dim * (i + 1), 0:self.__trajectory_interpolation.num_optimization_variables] = \
                self.__trajectory_interpolation.get_input_trajectory_vector_matrix(
                    [self.__trajectory_interpolation.breakpoints[-1]], derivative_order=i + 1)

        self.__A_eq[a + 2 * self.__dim:a + 3 * self.__dim,
            self.__trajectory_interpolation.num_optimization_variables-self.__dim:self.__trajectory_interpolation.num_optimization_variables] = \
            np.eye(self.__dim)

        # Inequality Constraints
        # position/speed/acc/jerk limit
        for i in range(0, 3):
            # x < ub
            self.__A_uneq_part[
            i * 2 * self.__dim * self.__num_state_constraint_sample_points:
            i * 2 * self.__dim * self.__num_state_constraint_sample_points +
            self.__dim * self.__num_state_constraint_sample_points, 0:self.__trajectory_interpolation.num_optimization_variables] = \
                self.input_to_trajectory_vector_matrix_state_wrapper(derivative=i)

            # lb < x <=> -x < -lb
            self.__A_uneq_part[
            i * 2 * self.__dim * self.__num_state_constraint_sample_points +
            self.__dim * self.__num_state_constraint_sample_points:
            (i + 1) * 2 * self.__dim * self.__num_state_constraint_sample_points, 0:self.__trajectory_interpolation.num_optimization_variables] = \
                -self.input_to_trajectory_vector_matrix_state_wrapper(derivative=i)

        # limit every optimization variable instead of only at the state constraint sample points
        a = self.__num_anti_collision_constraints
        b = self.__num_state_constraint_sample_points
        c = self.__trajectory_interpolation.num_optimization_variables
        n = 3 * 2 * self.__dim
        # Ax < b -> x < ub

        self.__A_uneq_part[n * b:n * b + c] = np.eye(c)
        self.__b_uneq_part[n * b:n * b + c] = np.tile(self.__upper_boundaries[3], self.__prediction_horizon)
        # lb < x -> -x < -lb
        self.__A_uneq_part[n * b + c:n * b + 2 * c] = -np.eye(c)
        self.__b_uneq_part[n * b + c:n * b + 2 * c] = - np.tile(self.__lower_boundaries[3],
                                                           self.__prediction_horizon)

        # limit weak constraint factors
        if self.__use_soft_constraints == 1:
            offset = self.__num_anti_collision_constraints + \
                3 * self.__dim * 2 * self.__num_state_constraint_sample_points + \
                2 * self.__trajectory_interpolation.num_optimization_variables
            self.__A_uneq[offset:offset+self.__num_anti_collision_constraints,
                self.__trajectory_interpolation.num_optimization_variables:] = np.eye(self.__num_anti_collision_constraints)
            self.__b_uneq[offset:offset + self.__num_anti_collision_constraints] = \
                np.zeros((self.__num_anti_collision_constraints, ))

            offset += self.__num_anti_collision_constraints
            self.__A_uneq[offset:offset + self.__num_anti_collision_constraints,
            self.__trajectory_interpolation.num_optimization_variables:] = -np.eye(self.__num_anti_collision_constraints)
            self.__b_uneq[offset:offset + self.__num_anti_collision_constraints] = \
                np.ones((self.__num_anti_collision_constraints,))*self.__options.weight_soft_constraint

    def calculate_trajectory(self, current_id,
                             current_state,
                             target_position,
                             index,
                             dynamic_trajectories,
                             static_trajectories,
                             previous_solution,
                             optimize_constraints,
                             dynamic_target_points,
                             static_target_points,
                             dynamic_coop_prio,
                             static_coop_prio,
                             dynamic_trajectory_age,
                             static_trajectory_age,
                             band_weights,
                             cooperative_objective_function=True,
                             use_nonlinear_mpc=False,
                             high_level_setpoints=None,
                             w_band=0.1,
                             ):
        """
        calculates data
        Parameters
        ----------
        current_state: np.array, shape(9,)
            current state of the agent
        target_position: np.array, shape(3,)
            target of the drone
        index:
            index of agent in dynamic_trajectories
        dynamic_trajectories:
            trajectories of agents that get recalculated too. Contains agent to recalculate at index
        static_trajectories:
            trajectories of agents that do not get recalculated or static obstacles.
        optimize_constraints: bool
            if constraints should be optimized
        previous_solution: TrajectoryCoefficients
            previous solution of optimization problem (in case the solver fails due to numerical issues)
        dynamic_target_points:
            target points of the agents in dynamic agents
        static_target_points:
            target points of the agents in static agents

        Returns
        -------
        optimal_coefficients: TrajectoryCoefficients
            if not valid, then the planned data will be returned
        """
        start = time.time()
        # build equality constraint right side
        a = self.__trajectory_interpolation.num_equality_constraints
        self.__b_eq[a: a + self.__dim] = - self.__state_trajectory_vector_matrix_dt @ current_state
        self.__b_eq[
        a + self.__dim: a + 2 * self.__dim] = - self.__state_trajectory_vector_matrix_dtdt @ current_state
        self.__b_eq[a + 2 * self.__dim: a + 3 * self.__dim] = np.zeros((self.__dim,))
        # inequality state constraints
        for i in range(0, 3):
            future_state = self.state_to_trajectory_vector_matrix_state_wrapper(derivative=i) @ current_state

            if i == 0:
                ub = np.tile(self.__upper_boundaries[i][current_id], self.__num_state_constraint_sample_points) - future_state
                lb = -np.tile(self.__lower_boundaries[i][current_id], self.__num_state_constraint_sample_points) + future_state

            else:
               ub = np.tile(self.__upper_boundaries[i], self.__num_state_constraint_sample_points) - future_state
               lb = -np.tile(self.__lower_boundaries[i], self.__num_state_constraint_sample_points) + future_state
            self.__b_uneq_part[
                i * 2 * self.__dim * self.__num_state_constraint_sample_points:
                i * 2 * self.__dim * self.__num_state_constraint_sample_points +
                self.__dim * self.__num_state_constraint_sample_points] = ub

            self.__b_uneq_part[i * 2 * self.__dim *
                self.__num_state_constraint_sample_points + self.__dim *
                self.__num_state_constraint_sample_points:
                (i + 1) * 2 * self.__dim *
                self.__num_state_constraint_sample_points] = lb

        b = 3 * self.__dim * 2 * self.__num_state_constraint_sample_points + \
            2 * self.__trajectory_interpolation.num_optimization_variables

        # build linear constraints (if nonlinear do not use static_trajectories)
        A_col, b_col = self.build_anti_collision_constraint_optimized(index, dynamic_trajectories,
                                                                      static_trajectories if not use_nonlinear_mpc else [],
                                                                      optimize_constraints, current_state,
                                                                      dynamic_target_points + static_target_points,
                                                                      dynamic_trajectory_age=dynamic_trajectory_age,
                                                                      static_trajectory_age=static_trajectory_age,
                                                                      )

        if len(A_col) != 0:
            A_uneq = np.concatenate((self.__A_uneq_part, A_col), axis=0)
            b_uneq = np.concatenate((self.__b_uneq_part, b_col), axis=0)
        else:
            A_uneq = self.__A_uneq_part
            b_uneq = self.__b_uneq_part

        # add weak constraints variables constraints
        num_weak_variables = len(dynamic_trajectories) + len(static_trajectories) - 1
        A_eq = self.__A_eq
        if num_weak_variables > 0:
            A_uneq_weak = np.zeros((len(A_uneq), num_weak_variables))
            A_uneq_weak[-num_weak_variables:, :] = np.eye(num_weak_variables)

            A_uneq = np.concatenate((A_uneq, A_uneq_weak), axis=1)

            A_eq = np.concatenate((self.__A_eq, np.zeros((len(self.__A_eq), num_weak_variables))), axis=1)

            # now add constraints for weak constraint variables
            A_uneq_weak = np.zeros((2*num_weak_variables, len(A_uneq[0])))
            A_uneq_weak[-2*num_weak_variables:-num_weak_variables, -num_weak_variables:] = np.eye(num_weak_variables)
            A_uneq_weak[-num_weak_variables:, -num_weak_variables:] = -np.eye(num_weak_variables)
            A_uneq = np.concatenate((A_uneq, A_uneq_weak), axis=0)

            b_uneq = np.concatenate(
                (b_uneq, np.array([w_band for i in range(num_weak_variables)]), np.zeros((num_weak_variables,))), axis=0)

        # build nonlinear constraints
        nonlinear_constraints = []
        linear_constraints = []
        if use_nonlinear_mpc:
            nonlinear_constraints = self.build_anti_collision_constraint_nonlinear(static_trajectories, current_state,
                                                                                   dynamic_trajectory_age=dynamic_trajectory_age,
                                                                                   static_trajectory_age=static_trajectory_age,)

            linear_constraints = [opt.LinearConstraint(A_uneq, lb=-np.inf, ub=b_uneq),
                                  opt.LinearConstraint(self.__A_eq, lb=self.__b_eq, ub=self.__b_eq)]

        # build optimization function
        future_pos = self.__state_to_trajectory_matrix_objective @ current_state

        new_target = np.zeros((3,))
        num_targets = 0
        target_changed = False
        only_change_target = False
        q = 0
        if cooperative_objective_function:
            for i in range(len(dynamic_trajectories)):
                if i == index:
                    continue
                other_agents_pos_list = [dynamic_trajectories[j][0] for j in range(len(dynamic_trajectories)) if
                                         j != index and j != i] + \
                                        [static_trajectories[j][0] for j in range(len(static_trajectories))]
                temp, new_target_temp = self.__calculate_cooperative_objective_function(dynamic_trajectories[index][-1],
                                                                     dynamic_target_points[index],
                                                                     dynamic_trajectories[i][-1],
                                                                     dynamic_target_points[i],
                                                                     dynamic_coop_prio[index],
                                                                     dynamic_coop_prio[i],
                                                                     other_agents_pos_list)
                if type(new_target_temp) != type(0):
                    target_changed = True
                    new_target += new_target_temp
                    num_targets += 1
                q += temp
            for i in range(len(static_trajectories)):
                other_agents_pos_list = [dynamic_trajectories[j][0] for j in range(len(dynamic_trajectories)) if
                                         j != index] + \
                                        [static_trajectories[j][0] for j in range(len(static_trajectories)) if j!=i]
                temp, new_target_temp = self.__calculate_cooperative_objective_function(dynamic_trajectories[index][-1],
                                                                     dynamic_target_points[index],
                                                                     static_trajectories[i][-1],
                                                                     static_target_points[i],
                                                                     dynamic_coop_prio[index],
                                                                     static_coop_prio[i],
                                                                     other_agents_pos_list
                                                                     )
                if type(new_target_temp) != type(0):
                    target_changed = True
                    new_target += new_target_temp
                    num_targets += 1
                q += temp
        real_target_position = target_position
        if target_changed:
            pass
            #new_target /= num_targets
            #real_target_position = new_target
        if only_change_target:
            q = - (np.tile(real_target_position, self.__num_objective_function_sample_points) - future_pos) @ self.__P
        else:
            if high_level_setpoints is not None:
                q += - (np.reshape(high_level_setpoints, (high_level_setpoints.size,)) - future_pos) @ self.__P
            else:
                q += - (np.tile(target_position, self.__num_objective_function_sample_points_pos) - future_pos) @ self.__P

        # now add weak constraints to objective function
        num_variables = num_weak_variables + self.__num_optimization_variables
        Q = np.zeros((num_variables, num_variables))
        Q[0:self.__num_optimization_variables, 0:self.__num_optimization_variables] = self.__Q
        # Q[self.__num_optimization_variables:, self.__num_optimization_variables:] = np.eye(num_weak_variables) * 0.1

        if num_weak_variables > 0:
            q = np.concatenate((q, np.zeros((num_weak_variables,))), axis=0)

        offset = 0
        for i in range(num_weak_variables):
            if i == index:
                offset = 1
            q[self.__num_optimization_variables+i] = -band_weights[i+offset] * w_band
            Q[self.__num_optimization_variables+i, self.__num_optimization_variables+i] = band_weights[i+offset]

        # solve optimization problem
        optimal_coefficients = None
        #print(f"Prep Time {time.time()-start}")
        if use_nonlinear_mpc:
            fun = lambda x: x.T @ self.__Q @ x + 2*q.T @ x
            fun_jacob = lambda x: 2* self.__Q @ x + 2*q
            options = {"maxiter": 1e2, "disp": False}
            result = opt.minimize(fun, x0=np.copy(previous_solution.flatten()), jac=fun_jacob,
                                  constraints=nonlinear_constraints + linear_constraints, method="SLSQP", options=options)
            if not result.success:
                print("No Solution Found for nonlinear solver")
                return TrajectoryCoefficients(None, False, previous_solution)
            print(result.message)
            optimal_coefficients = copy.deepcopy(result.x)
            future_pos_from_current_state = self.__state_to_trajectory_matrix_collision @ current_state
            for i in range(len(static_trajectories)):
                trajectory = static_trajectories[i, :, :]  # this evaluation has to be done before the lambda, otherwise i is evaluated at end of loop (late binding)
                fun = lambda x: np.linalg.norm(self.__downwash_scaling @ (np.reshape(
                    self.__coefficients_to_trajectory_matrix_collision @ x + future_pos_from_current_state,
                    newshape=trajectory.shape) - trajectory).T,
                                               axis=0) ** 2

                """print(fun(optimal_coefficients))
                print(
                    np.reshape(
                        self.__coefficients_to_trajectory_matrix_collision @ optimal_coefficients + future_pos_from_current_state,
                        newshape=static_trajectories[i, :, :].shape) - static_trajectories[i, :, :])
                print(static_trajectories[i, :, :])
                print(self.__options.r_min**2)
                print(fun(previous_solution.flatten())>self.__options.r_min**2)
                print(fun(optimal_coefficients.flatten()) > self.__options.r_min ** 2)
                print(len(nonlinear_constraints))
                print("xxxxxxxxxxx")
                print(self.__A_eq @ result.x - self.__b_eq)"""
        elif self.__options.use_qpsolvers:
            start = time.time()
            optimal_coefficients = qp.solve_qp(P=Q,
                                               q=q,
                                               G=A_uneq, h=b_uneq,
                                               A=A_eq, b=self.__b_eq, solver='quadprog')
            #print(time.time()-start)
            if optimal_coefficients is None:
                print("No Solution Found")
                return TrajectoryCoefficients(None, False, previous_solution)
        else:
            P = matrix(self.__Q)
            q = matrix(q)
            G = matrix(A_uneq)
            h = matrix(b_uneq)
            A = matrix(self.__A_eq)
            b = matrix(self.__b_eq)

            try:
                optimal_coefficients = solvers.qp(P, q, G, h, A, b)
                successful = optimal_coefficients['status'] == 'optimal'
                optimal_coefficients = optimal_coefficients['x']
                optimal_coefficients = np.squeeze(np.asarray(optimal_coefficients))
            except ValueError:
                successful = False

            # print('Optimization Done')
            if not successful:
                print("No Solution Found")
                return TrajectoryCoefficients(None, False, previous_solution)


        # reshaped: optimale Koeffizienten auf die prediction horizons aufteilen
        # print("------------------")
        # print(optimal_coefficients[-num_weak_variables:])
        reshaped = np.reshape(optimal_coefficients[0:self.__trajectory_interpolation.num_optimization_variables],
                              (self.prediction_horizon, self.__dim))

        return_value = TrajectoryCoefficients(copy.deepcopy(reshaped),
                                              True, previous_solution)

        """
        drone_breakpoints = np.linspace(0, self.__breakpoints[-1], (len(self.__breakpoints) - 1) * 5 + 1)
        pos = self.__trajectory_interpolation.interpolate_vector(drone_breakpoints, return_value, derivative_order=0,
                                                                 x0=current_state)
        vel = self.__trajectory_interpolation.interpolate_vector(drone_breakpoints, return_value, derivative_order=1,
                                                                 x0=current_state)
        acc = self.__trajectory_interpolation.interpolate_vector(drone_breakpoints, return_value, derivative_order=2,
                                                                 x0=current_state)
        """
        # print(pos[:20])

        """
        if previous_solution.valid:
            P = self.__Q
            G = self.__A_uneq
            h = self.__b_uneq
            A = self.__A_eq
            b = self.__b_eq

            Lambda = self.__trajectory_interpolation.get_input_trajectory_vector_matrix(self.__breakpoints,
                                                                                        derivative_order=0)
            A_0 = self.__trajectory_interpolation.get_state_trajectory_vector_matrix(self.__breakpoints,
                                                                                     derivative_order=0)
            mdict = {"P": P, "G": G, "q": q, "h": h, "A": A, "b": b, "Lambda": Lambda, "A_0": A_0, "x0": current_state}
            savemat("QP-Problem_full.mat", mdict)

        """
        # if drone_id == 0:
        # print('Optimization Input at timestep ' + str(timestep) + 's :')
        # print(reshaped[:2])
        return return_value

    def build_anti_collision_constraint_nonlinear(self, static_trajectories, current_state, dynamic_trajectory_age, static_trajectory_age):
        constraints = []
        future_pos_from_current_state = self.__state_to_trajectory_matrix_collision @ current_state
        for i in range(len(static_trajectories)):
            constraints.append(opt.NonlinearConstraint(self.late_binding_avoider_constraint(static_trajectories, future_pos_from_current_state, i, trajectory_age=static_trajectory_age[i]), lb=self.__options.r_min**2, ub=np.inf, keep_feasible=False,
                                                       jac=self.late_binding_avoider_constraint_jacobian(static_trajectories, future_pos_from_current_state, i, trajectory_age=static_trajectory_age[i])))
        return constraints

    def late_binding_avoider_constraint(self, static_trajectories, future_pos_from_current_state, i, trajectory_age):
        trajectory = static_trajectories[i, :, :]  # this evaluation has to be done before the lambda, otherwise i is evaluated at end of loop (late binding)
        scaling = np.kron(np.eye(len(trajectory), dtype=int), self.__downwash_scaling)
        for j in range(len(trajectory)):
            if j >= len(trajectory) - trajectory_age - 2:
                scaling[j*3 + 2] = 0
        #fun = lambda x: np.linalg.norm(self.__downwash_scaling @ (
        #            np.reshape(self.__coefficients_to_trajectory_matrix_collision @ x + future_pos_from_current_state,
        #                       newshape=trajectory.shape) - trajectory).T, axis=0) ** 2
        fun = lambda x: np.linalg.norm(self.__downwash_scaling @ np.reshape(scaling @ (self.__coefficients_to_trajectory_matrix_collision @ x + future_pos_from_current_state
                            - np.ndarray.flatten(trajectory)), newshape=trajectory.shape).T, axis=0) ** 2
        return fun

    def late_binding_avoider_constraint_jacobian(self, static_trajectories, future_pos_from_current_state, i, trajectory_age):
        trajectory = static_trajectories[i, :, :]  # this evaluation has to be done before the lambda, otherwise i is evaluated at end of loop (late binding)

        fun = lambda x: [self.late_binding_avoider_single_constraint_jacobian(x, trajectory, future_pos_from_current_state, j, scaling=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) if (j >= len(trajectory) - trajectory_age - 2) else self.__downwash_scaling)
                         for j in range(len(trajectory))]
        return fun

    def late_binding_avoider_single_constraint_jacobian(self, x, trajectory, future_pos_from_current_state, j, scaling):
        return (self.__coefficients_to_trajectory_matrix_collision).T[:, j*3:(j+1)*3] @ \
                        scaling @ scaling @ \
               ((self.__coefficients_to_trajectory_matrix_collision @ x + future_pos_from_current_state)[j*3:(j+1)*3] - trajectory[j, :])

    def build_anti_collision_constraint(self, planned_trajectory, other_trajectory, current_state, target_position,
                                        other_target, other_id, drone_id):
        """build anti collision constraint A*coefficients <= b for one data

        Parameters
        ----------
            planned_trajectory: np.array, shape(., 3)
                data that was planned a timestep ago. Should start with the first data point to optimize.
            other_trajectory: np.array, shape(., 3)
                trajectories of an other agent.
                Should start with the first data point to optimize.
            current_state: np.array, shape(3,)
                state of the drone at the beginning of the prediction horizon
            target_position: np.array, shape(3,)
                target position of the drone

        Returns
        -------
            A: np.array, shape(., number optimization parameters)
                anti collision constraint matrix
            b: np.array, shape(.,)
                right side
        """
        min_angle = 20.0 * math.pi / 180  # min angle in radians
        downwash_coefficient = self.__options.downwash_scaling_factor
        downwash_scaling = np.diag([1, 1, 1.0 / float(downwash_coefficient)])
        A = np.zeros((planned_trajectory.shape[0], self.__num_optimization_variables))
        b = np.zeros((planned_trajectory.shape[0],))
        future_pos_from_current_state = self.__state_to_trajectory_matrix_collision @ current_state
        future_pos_from_current_state = np.reshape(future_pos_from_current_state,
                                                   (planned_trajectory.shape[0], self.__dim))
        for i in range(0, planned_trajectory.shape[0]):
            # build normal vector
            relative_trajectory = (planned_trajectory[i] - other_trajectory[i]) @ downwash_scaling
            own_dist_to_target = (target_position - planned_trajectory[i]) @ downwash_scaling
            other_dist_to_target = (other_target - other_trajectory[i]) @ downwash_scaling

            if self.__options.skewed_plane_BVC:
                normal_vector_bvc = self.calc_BVC_normal_vector(relative_trajectory, own_dist_to_target,
                                                            other_dist_to_target, min_angle, own_id=drone_id,
                                                            other_id=other_id)
            else:
                normal_vector_bvc = relative_trajectory

            distance = np.linalg.norm(normal_vector_bvc)
            n_0 = normal_vector_bvc / distance

            if abs(n_0 @ ((planned_trajectory[i] - other_trajectory[
                i]) @ downwash_scaling)) < self.__options.r_min - 0.001:  # -0.001 to ignore numerical errors
                #print(f'Collision!!: {abs(n_0 @ ((planned_trajectory[i] - other_trajectory[i]) @ downwash_scaling))}')
                n_0 = relative_trajectory / np.linalg.norm(relative_trajectory)  # use non-skewed plane in case of collision


            # right side
            b[i] = -(self.__options.r_min + self.__use_soft_constraints*self.__options.soft_constraint_max +
                     distance*self.__guarantee_anti_collision) / 2.0 - np.dot(n_0, (other_trajectory[i] -
                                                                           future_pos_from_current_state[
                                                                               i]) @ downwash_scaling)

            # left side
            A[i, 0:self.__trajectory_interpolation.num_optimization_variables] = \
                -n_0 @ downwash_scaling @ self.__coefficients_to_trajectory_matrix_collision[
                i * self.__dim:(i + 1) * self.__dim, :]
            if self.__use_soft_constraints == 1:
                A[i, self.__trajectory_interpolation.num_optimization_variables + i] = 1

        return A, b

    def build_anti_collision_constraint_optimized(self, index, dynamic_trajectories, static_trajectories,
                                                  optimize_constraints, current_state, target_points,
                                                  dynamic_trajectory_age,
                                                  static_trajectory_age,
                                                  only_static=True, use_complex_priority=False):
        """build anti collision constraint A*coefficients <= b for one data and removes redudant constraints

        Parameters
        ----------
            index:
                index of agent in dynamic_trajectories
            dynamic_trajectories: np.array, shape(., ., 3)
                trajectories of agents that get recalculated too. Contains agent to recalculate at index
            static_trajectories: np.array, shape(., ., 3)
                trajectories of agents that do not get recalculated or static obstacles.
            optimize_constraints: bool
                true if optimize constraints

        Returns
        -------
            A: np.array, shape(., number optimization parameters)
                anti collision constraint matrix
            b: np.array, shape(.,)
                right side
        """

        A_list = []
        b_list = []
        if len(dynamic_trajectories) == 0:
            return A_list, b_list
        future_pos_from_current_state = self.__state_to_trajectory_matrix_collision @ current_state
        future_pos_from_current_state = np.reshape(future_pos_from_current_state,
                                                   (dynamic_trajectories[0].shape[0], self.__dim))

        # go through time and remove redundant constraints
        for i in range(0, dynamic_trajectories.shape[1]):
            if only_static:
                # -2, because the last point should alwas be vertical
                dynamic_points_use_vertical_constraints = [i >= dynamic_trajectories.shape[1] - dynamic_trajectory_age[j] - 2 for j in range(dynamic_trajectories.shape[0])]
                static_points_use_vertical_constraints = [i >= dynamic_trajectories.shape[1] - static_trajectory_age[j] - 2 for j in range(static_trajectories.shape[0])]if not len(static_trajectories) == 0 else None
                assert not optimize_constraints
                points_on_plane, normal_vectors = self.remove_redundant_planes_only_static(
                    index, dynamic_trajectories[:, i, :],
                    static_trajectories[:, i, :] if not len(static_trajectories) == 0 else None,
                    optimize_constraints, use_complex_priority, target_points=target_points,
                    dynamic_points_use_vertical_constraints=dynamic_points_use_vertical_constraints,
                    static_points_use_vertical_constraints=static_points_use_vertical_constraints)
            else:
                assert False
                points_on_plane, normal_vectors = self.remove_redundant_planes(index, dynamic_trajectories[:, i, :],
                                                                               static_trajectories[:, i, :],
                                                                               optimize_constraints)
            for j in range(len(points_on_plane)):
                A_list.append(- normal_vectors[j] @ self.__downwash_scaling @
                              self.__coefficients_to_trajectory_matrix_collision[i * self.__dim:(i + 1) * self.__dim, :])
                b_list.append(-np.dot(normal_vectors[j],
                                      points_on_plane[j] - self.__downwash_scaling @ future_pos_from_current_state[i]))


        self.__last_num_constraints = len(b_list)
        return np.array(A_list), np.array(b_list)

    def remove_redundant_planes(self, index, dynamic_points, static_points, optimize_constraints):
        """
        removes redundant planes for BVC
        Parameters
        ----------
        index:
            index of corresponding agent
        dynamic_points:
            points the data should keep r_min distance too
        static_points:
            points the data should keep r_min distance too
        optimize_constraints: bool
            if constraints should be optimized

        Returns
        ------
            points_on_plane: points on the planes (one point per plane)
            normal_vectors: normal vector of the corresponding planes
        """
        points_on_plane = []
        normal_vectors = []
        dynamic_agents_constraints = [None for i in range(len(dynamic_points)**2)]  # this represenst the constraints between two dynamic agents, which already have been calculated.
        for dynamic_ind in range(len(dynamic_points)):
            planned_point = self.__downwash_scaling @ dynamic_points[dynamic_ind]
            # recursively remove redundant planes
            obstacles_points = np.array([self.__downwash_scaling @ dynamic_points[j] for j in range(len(dynamic_points))] +
                                        [self.__downwash_scaling @ static_points[j] for j in range(len(static_points))])

            planes_normal_vectors = np.array([self.calculate_normal_vector(planned_point, p)
                                                     for p in obstacles_points])
            planes_dists = np.array([np.linalg.norm(v) for v in planes_normal_vectors])

            for dist in planes_dists:
                if dist < self.__options.r_min and dist != 0:
                    print('Collision!!')
                    print(dist)
                    pass
                else:
                    print("No")

            ind_sorted_planes = np.argsort(planes_dists)
            already_used = [False for i in range(len(obstacles_points))]
            while True:
                i = 1  # i = 0 for dynamic_ind == ind_sorted_planes[0]
                while i < len(already_used):
                    if not already_used[ind_sorted_planes[i]]:
                        break
                    i += 1
                # all are already put into constraints
                if i == len(already_used):
                    break
                index_current_point = ind_sorted_planes[i]
                distance = planes_dists[index_current_point]
                n_0 = planes_normal_vectors[index_current_point] / distance
                n_0_temp = n_0
                point_temp = obstacles_points[index_current_point] + \
                                (planned_point - obstacles_points[index_current_point]) / 2 + \
                                n_0 * self.__options.r_min / 2

                already_added = False
                if index_current_point < len(dynamic_points):
                    # if there already exist constraints, use them, otherwise use new BVC
                    if dynamic_agents_constraints[dynamic_ind + index_current_point*len(dynamic_points)] is None:
                        point = obstacles_points[index_current_point] + \
                                (planned_point - obstacles_points[index_current_point]) / 2 + \
                                n_0 * self.__options.r_min / 2
                        dynamic_agents_constraints[dynamic_ind + index_current_point * len(dynamic_points)] = (point, n_0)
                        dynamic_agents_constraints[index_current_point + dynamic_ind * len(dynamic_points)] = \
                            (point - n_0*self.__options.r_min, -n_0)

                        if index_current_point == index:
                            points_on_plane.append(point - n_0 * self.__options.r_min)
                            normal_vectors.append(-n_0)
                    else:
                        point = dynamic_agents_constraints[dynamic_ind + index_current_point * len(dynamic_points)][0]
                        n_0 = dynamic_agents_constraints[dynamic_ind + index_current_point * len(dynamic_points)][1]
                        already_added = True
                        if np.dot(n_0, planned_point - point) < 0:
                            if np.linalg.norm(point_temp - point) > 0.01:

                                print("asdfsdfsdfdfdfdfdfdfdfff")
                                print(np.dot(n_0, planned_point - point))
                                print(n_0)
                                print(n_0_temp)
                                print(point)
                                print(point_temp)

                                print(obstacles_points[index_current_point])
                                print(obstacles_points[dynamic_ind])
                                print(np.linalg.norm(obstacles_points[index_current_point] - obstacles_points[dynamic_ind]))
                else:
                    point = obstacles_points[index_current_point] + n_0 * self.__options.r_min

                # if index is equal to current calculated index, put constraints into list
                if dynamic_ind == index and not already_added:
                    points_on_plane.append(point)
                    normal_vectors.append(n_0)
                already_used[index_current_point] = True

                if not optimize_constraints:
                    continue

                # now check if some constraints are redundant
                i += 1
                while i < len(already_used):
                    if not already_used[ind_sorted_planes[i]]:
                        # print(np.dot(n_0, (static_obstacles_points[i] - point)) )
                        # if the current watched point is on the left site and with distance of r_min, there is no
                        # need for an additional constraint
                        if np.dot(n_0,
                                  obstacles_points[ind_sorted_planes[i]] - point) < -self.__options.r_min:
                            already_used[ind_sorted_planes[i]] = True

                            # if it is an dynamic point, this means, that it has to take this constraint too.
                            if ind_sorted_planes[i] < len(dynamic_points):
                                already_used[ind_sorted_planes[i]] = False
                                if dynamic_agents_constraints[dynamic_ind + ind_sorted_planes[i] * len(dynamic_points)] is None:
                                    #print("Hello thereeeeee")
                                    #already_used[ind_sorted_planes[i]] = False
                                    dynamic_agents_constraints[dynamic_ind + ind_sorted_planes[i] * len(dynamic_points)] = (
                                    point, n_0)
                                    dynamic_agents_constraints[ind_sorted_planes[i] + dynamic_ind * len(dynamic_points)] = \
                                        (point - n_0 * self.__options.r_min, -n_0)
                                    already_used[ind_sorted_planes[i]] = True

                                    if ind_sorted_planes[i] == index:
                                        points_on_plane.append(point - n_0 * self.__options.r_min)
                                        normal_vectors.append(-n_0)


                    i += 1


        """points_on_plane = []
        normal_vectors = []

        for i in range(other_points.shape[0]):
            if dynamic_obstacles[i]:
                n_0 = self.calculate_normal_vector(planned_point, other_points[i])
                distance = np.linalg.norm(n_0)
                n_0 = n_0 / distance
                if abs(n_0 @ ((planned_point - other_points[i]) @ self.__downwash_scaling)) \
                        < self.__options.r_min - 0.001:  # -0.001 to ignore numerical errors
                    print('Collision!!')
                    print(abs(n_0 @ ((planned_point - other_points[i]) @ self.__downwash_scaling)))
                points_on_plane.append(other_points[i] @ self.__downwash_scaling+ n_0*(distance+self.__options.r_min) / 2)
                normal_vectors.append(n_0)

        # recursively remove redundant planes
        static_obstacles_points = np.array([other_points[i]
                                            for i in range(len(other_points)) if not dynamic_obstacles[i]])
        static_planes_normal_vectors = np.array([self.calculate_normal_vector(planned_point, p)
                                                 for p in static_obstacles_points])
        static_planes_dists = np.array([np.linalg.norm(v) for v in static_planes_normal_vectors])

        ind_sorted_planes = np.argsort(static_planes_dists)
        already_used = [False for i in range(len(static_obstacles_points))]
        while True:
            i = 0
            while i < len(already_used):
                if not already_used[i]:
                    break
                i += 1
            # all are already put into constraints
            if i == len(already_used):
                break

            distance = np.linalg.norm(static_planes_normal_vectors[i])
            n_0 = static_planes_normal_vectors[i] / distance
            point = self.__downwash_scaling @ static_obstacles_points[i] + n_0 * self.__options.r_min
            points_on_plane.append(point)
            normal_vectors.append(n_0)
            already_used[i] = True

            i += 1
            while i < len(already_used):
                if not already_used[i]:
                    #print(np.dot(n_0, (static_obstacles_points[i] - point) @ self.__downwash_scaling) )
                    if np.dot(n_0, static_obstacles_points[i] @ self.__downwash_scaling - point) < -self.__options.r_min:
                        already_used[i] = True
                i += 1"""

        return points_on_plane, normal_vectors

    def remove_redundant_planes_only_static(self, index, dynamic_points, static_points, optimize_constraints,
                                            use_complex_priority, target_points, dynamic_points_use_vertical_constraints,
                                            static_points_use_vertical_constraints, optimal_removal=False):
        """
        removes redundant planes for static BVC
        Parameters
        ----------
        index:
            index of corresponding agent
        dynamic_points:
            points the data should keep r_min distance too
        static_points:
            points the data should keep r_min distance too
        optimize_constraints: bool
            if constraints should be optimized
        dynamic_points_use_vertical_constraints: list(bool)
        static_points_use_vertical_constraints: list(bool)
        optimal_removal: bool
            if the optimal removal should be used

        Returns
        ------
            points_on_plane: points on the planes (one point per plane)
            normal_vectors: normal vector of the corresponding planes
        """
        points_on_plane = []
        normal_vectors = []
        dynamic_agents_constraints = [None for i in range(len(dynamic_points)**2)]  # this represenst the constraints between two dynamic agents, which already have been calculated.

        planned_point = self.__downwash_scaling @ dynamic_points[index]

        # add dynamic agents (do not remove redundant constraints)
        obstacles_points = np.array([self.__downwash_scaling @ dynamic_points[j] for j in range(len(dynamic_points))])
        planes_normal_vectors = np.array([self.calculate_normal_vector(planned_point, obstacles_points[j],
                                                                       dynamic_points_use_vertical_constraints[j])
                                          for j in range(len(obstacles_points))])
        planes_dists = np.array([np.linalg.norm(v) for v in planes_normal_vectors])

        for dist in planes_dists:
            if dist < self.__options.r_min and dist != 0:
                print('Collision!!')
                print(dist)
                pass

        for i in range(len(obstacles_points)):
            if i == index:
                continue
            n_0 = planes_normal_vectors[i] / planes_dists[i]
            point = (obstacles_points[i] + planned_point) / 2 + n_0*self.__options.r_min / 2
            points_on_plane.append(point)
            normal_vectors.append(n_0)
        if static_points is None:
            return points_on_plane, normal_vectors
        # recursively remove redundant planes
        # first remove planes, which already are satisfied because of the dynamic ones
        obstacles_points = np.array([self.__downwash_scaling @ static_points[j] for j in range(len(static_points))])

        planes_normal_vectors = np.array([self.calculate_normal_vector(planned_point, obstacles_points[j],
                                                                       static_points_use_vertical_constraints[j])
                                          for j in range(len(obstacles_points))])
        planes_dists = np.array([np.linalg.norm(v) for v in planes_normal_vectors])

        if not use_complex_priority:
            planes_rev_prios = planes_dists
        else:
            planes_rev_prios = [self.__calculate_rev_priority(planned_point, p, target_points[index])
                                for p in obstacles_points]

        for dist in planes_dists:
            if dist < self.__options.r_min and dist != 0:
                #print('Collision!!')
                #print(dist)
                pass

        ind_sorted_planes = np.argsort(planes_rev_prios)
        already_used = [False for _ in range(len(obstacles_points))]

        if optimize_constraints:
            for j in range(len(already_used)):
                for k in range(len(points_on_plane)):
                    if np.dot(normal_vectors[k],
                              obstacles_points[j] - points_on_plane[k]) < -self.__options.r_min:
                        already_used[j] = True
                        break

        assert not optimal_removal, "Not implemented correctly"

        j = 0
        while True:
            i = 0
            if optimal_removal:
                i = j
                finished = True
                for a in range(j, len(already_used)):
                    if not already_used[ind_sorted_planes[a]]:
                        finished = False
                        break
                already_used[ind_sorted_planes[j]] = True
                if finished:
                    break
            else:
                while i < len(already_used):
                    if not already_used[ind_sorted_planes[i]]:
                        break
                    i += 1
            # all are already put into constraints
            if i == len(already_used):
                break
            index_current_point = ind_sorted_planes[i]
            distance = planes_dists[index_current_point]
            n_0 = planes_normal_vectors[index_current_point] / (distance+1e-9)
            point = obstacles_points[index_current_point] + n_0 * self.__options.r_min

            points_on_plane.append(point)
            normal_vectors.append(n_0)
            already_used[index_current_point] = True

            if not optimize_constraints:
                continue

            # now check if some constraints are redundant
            i += 1
            while i < len(already_used):
                if not already_used[ind_sorted_planes[i]]:
                    # print(np.dot(n_0, (static_obstacles_points[i] - point)) )
                    # if the current watched point is on the left site and with distance of r_min, there is no
                    # need for an additional constraint
                    if np.dot(n_0,
                              obstacles_points[ind_sorted_planes[i]] - point) < -self.__options.r_min:
                        already_used[ind_sorted_planes[i]] = True
                i += 1

            j += 1

        return points_on_plane, normal_vectors

    def __calculate_cooperative_objective_function(self, pos, target_pos, other_agent_pos, other_agent_target_pos,
                                                   prio, other_agent_prio, other_agents_pos_list):
        #print(prio)
        if prio >= other_agent_prio:
            return 0, 0
        # if the distances are very close, do not use the cooperative function
        #if (not (np.linalg.norm(other_agent_pos - other_agent_target_pos) > np.linalg.norm(pos - target_pos) +
        #        self.__options.min_distance_cooperative*0)) or np.linalg.norm(other_agent_pos - other_agent_target_pos) < self.__options.r_min:
        #    return 0
        if np.linalg.norm(other_agent_pos - other_agent_target_pos) < self.__options.r_min:
            print(other_agent_target_pos)
            print(other_agent_pos)
            print("Hello there1")
            return 0

        a = np.cross(pos - other_agent_pos, other_agent_target_pos - other_agent_pos)
        normal_vector = np.cross(other_agent_target_pos - other_agent_pos, a)
        normal_vector /= np.linalg.norm(normal_vector) + 1e-9

        # if agent is not in between target and pos do not dodge (no need to)
        distance_other_agent_to_target = np.linalg.norm(other_agent_pos - other_agent_target_pos)
        if np.dot(pos - other_agent_pos, other_agent_target_pos - other_agent_pos) < 0 or \
                np.dot(pos - other_agent_pos, other_agent_target_pos - other_agent_pos) > distance_other_agent_to_target:
            return 0, 0
        # if agent is too far away or agent is not in between target and pos do not dodge (no need to)
        distance_to_straight_path = np.dot(normal_vector, pos - other_agent_pos)
        #if distance_to_straight_path > self.__options.r_min:
        #    return 0

        # if there are other agents between line and this agent, get away from them and not the line.
        normal_vector_straight_path = (other_agent_target_pos - other_agent_pos) / distance_other_agent_to_target
        dodging_distance = distance_to_straight_path
        dodging_pos = pos - normal_vector * dodging_distance
        for p in other_agents_pos_list:
            if abs(np.dot(p - pos, normal_vector_straight_path)) < self.__options.r_min and \
                    0 < np.dot(pos-p, normal_vector) < 2*self.__options.r_min:
                if dodging_distance > np.dot(pos-p, normal_vector):
                    dodging_distance = np.dot(pos-p, normal_vector)
                    dodging_pos = p

        normal_vector += 2*(np.random.rand(3)-0.5)*self.__normal_vector_noise
        dodging_pos_dist_to_line = np.dot(dodging_pos - other_agent_pos, normal_vector)
        dist_to_keep = self.__options.r_min*1.2
        # do nothing
        if dodging_distance > dist_to_keep:
            return -self.__options.weight_cooperative / (dodging_distance + 0.1) * \
                   np.tile(normal_vector @ self.__downwash_scaling, self.__num_objective_function_sample_points) \
                   @ self.__coefficients_to_trajectory_matrix_state, 0
        new_target_vector = pos + normal_vector * (dist_to_keep - dodging_distance)

        normal_vector += 2*(np.random.rand(3)-0.5)*1.0

        return -self.__options.weight_cooperative / (dodging_distance + 0.1) *\
               np.tile(normal_vector @ self.__downwash_scaling, self.__num_objective_function_sample_points) \
               @ self.__coefficients_to_trajectory_matrix_state, new_target_vector

    def calculate_normal_vector(self, planned_point, other_point, use_vertical_constraint):
        n = planned_point - other_point
        if use_vertical_constraint and False:
            n[2] = 0
        return n

    def input_to_trajectory_vector_matrix_state_wrapper(self, derivative):
        """ returns the desired data vector matrix for state constraint sample points"""

        if derivative == 0:
            return self.__coefficients_to_trajectory_matrix_state
        elif derivative == 1:
            return self.__coefficients_to_trajectory_derivative_1_matrix_state
        elif derivative == 2:
            return self.__coefficients_to_trajectory_derivative_2_matrix_state
        else:
            return None

    def state_to_trajectory_vector_matrix_state_wrapper(self, derivative):
        """ returns the desired data vector matrix for state constraint sample points"""

        if derivative == 0:
            return self.__state_to_trajectory_matrix_state
        elif derivative == 1:
            return self.__state_to_trajectory_derivative_1_matrix_state
        elif derivative == 2:
            return self.__state_to_trajectory_derivative_2_matrix_state
        else:
            return None

    def calc_BVC_normal_vector(self, relative_trajectory, own_dist_to_target, other_dist_to_target, min_angle, own_id,
                               other_id):
        """ calculates the normal vector that should be used for the BVC of the anti collision constraint

        Parameters:
            own_position: np.array, shape(3,)
                position of current anti collision sample point of the own planned data
            other_position: np.array, shape(3,)
                position of current anti collision sample point of the other drone's planned data
            own_target: np.array, shape(3,)
                own target position
            other_target: np.array, shape(3,)
                other drone's target position
            """
        own_dist = np.linalg.norm(own_dist_to_target)
        other_dist = np.linalg.norm(other_dist_to_target)
        tol = 0.1
        if (own_dist - other_dist) > tol:
            bvc_normal_vector = -self.rotate_plane_outside_cone(own_dist_to_target, min_angle, -relative_trajectory,
                                                                positive_rotation=True)
        elif (own_dist - other_dist) < -tol:
            bvc_normal_vector = self.rotate_plane_outside_cone(other_dist_to_target, min_angle, relative_trajectory,
                                                               positive_rotation=True)
        else:
            if own_id > other_id:
                bvc_normal_vector = -self.rotate_plane_outside_cone(own_dist_to_target, min_angle, -relative_trajectory,
                                                                    positive_rotation=True)
            else:
                bvc_normal_vector = self.rotate_plane_outside_cone(other_dist_to_target, min_angle, relative_trajectory,
                                                                   positive_rotation=True)
        return bvc_normal_vector

    def rotate_plane_outside_cone(self, cone_axis, cone_angle, plane_normal_vector, positive_rotation=True):
        """ Rotates the normal vector of a plane so that it is outside of a cone defined by the cone_axis and the
        cone_angle

        Parameters:
            cone_axis: np.array, shape(3,)
                central axis of the cone
            cone_angle: float
                opening angle of the cone's tip
            plane_normal_vector: np.array, shape(3,)
                normal vector of the plane that should be rotated
            positive_rotation: bool
                choose, whether the vector should be rotated in the mathematically positive (right hand) direction
            """

        angle = self.angle_between(cone_axis, plane_normal_vector)
        if angle == 0:
            plane_normal_vector += np.array([0, 0, 0.01])  # if vectors are parallel, rotate plane vector up
            angle = self.angle_between(cone_axis, plane_normal_vector)
        vn = np.cross(cone_axis, plane_normal_vector)

        angle_difference = max((cone_angle - angle, 0))
        if not positive_rotation:
            angle_difference *= -1
        plane_normal_vector = self.rotate_vector(vn, angle_difference, plane_normal_vector)
        return plane_normal_vector

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return np.divide(vector, 1.0 * np.linalg.norm(vector))

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotate_vector(self, rot_axis, theta, vector):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        if theta != 0:
            rot_axis = np.asarray(self.unit_vector(rot_axis))
            a = math.cos(theta / 2.0)
            b, c, d = -rot_axis * math.sin(theta / 2.0)
            aa, bb, cc, dd = a * a, b * b, c * c, d * d
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            rot_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                   [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                   [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
            return rot_matrix @ vector
        else:
            return vector

    @property
    def last_num_constraints(self):
        return self.__last_num_constraints

    @property
    def prediction_horizon(self):
        """

        Returns
        -------
            prediction_horizon: int
                length of calculate data in number of timesteps
        """
        return self.__prediction_horizon
