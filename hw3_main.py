'''
ECE276A WI21 PR3: Visual-Inertial SLAM
'''

from utils import *
from scipy.linalg import expm, inv

### Visual-Inertial SLAM

if __name__ == '__main__':

    ### Load Sensor Data

    # data 1. this data has features, use this if you plan to skip the extra credit feature detection and tracking part
    filename = "./data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename, load_features=True)

    print(np.shape(t))                 # time: (1,3026)
    print(np.shape(features))          # (4,13289,3026)
    print(np.shape(linear_velocity))   # (3,3026)
    print(np.shape(angular_velocity))  # (3,3026)
    print(np.shape(K))                 # intrinsic calibration: (3,3)
    print(np.shape(b))                 # baseline: (1,1)
    print(np.shape(imu_T_cam))         # extrinsic calibration: (4,4)

    landmark_count_og = np.shape(features)[1]       # 13289
    transform_imu2camera = inv(imu_T_cam)

    # downsample landmarks
    landmarks = features[:, 0:landmark_count_og:10, :]
    feature_count = np.shape(landmarks)[2]          # 3026
    landmark_count = np.shape(landmarks)[1]         # 1329

    ### Initialize Parameters

    V = 90  # variance for gaussian noise

    # landmark mean and covariance (lecture 13, slide 8)
    mu_landmarks = -1 * np.ones((4, landmark_count))  # initialize landmarks as empty
    sigma_landmarks = np.identity(3*landmark_count) * V

    temp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    initialization = np.kron(np.identity(landmark_count), temp)

    # stereo camera calibration matrix M (lecture 13, slide 2)
    f_su = K[0][0]
    f_sv = K[1][1]
    c_u = K[0][2]
    c_v = K[1][2]

    M = np.array([[f_su, 0, c_u, 0],
                  [0, f_sv, c_v, 0],
                  [f_su, 0, c_u, -f_su * b],
                  [0, f_sv, c_v, 0]])

    # initial imu mean and variance
    mu_imu = np.identity(4)
    sigma_imu = np.identity(6)

    # initial imu trajectory over all time
    all_poses_size = (4, 4, feature_count)
    trajectory_imu = np.zeros(all_poses_size)
    trajectory_imu[:,:,0] = mu_imu  # pose at timestamp 0 initialized with identity matrix

    linear_velocity = np.transpose(linear_velocity)
    angular_velocity = np.transpose(angular_velocity)

    for i in range(feature_count):

        print('iteration:', i)

        ### a) IMU-based Localization via EKF Prediction

        # stack velocities (lecture 13, slide 13)
        v_t = linear_velocity[i,:]
        w_t = angular_velocity[i,:]
        u_t = np.vstack((v_t, w_t))

        # velocity hats (lecture 13, slide 15)
        v_t_hat = np.array([[0, -v_t[2], v_t[1]], [v_t[2], 0, -v_t[0]], [-v_t[1], v_t[0], 0]])
        w_t_hat = np.array([[0, -w_t[2], w_t[1]], [w_t[2], 0, -w_t[0]], [-w_t[1], w_t[0], 0]])

        u_t_hat = np.array([[w_t_hat[0][0], w_t_hat[0][1], w_t_hat[0][2], v_t[0]],
                            [w_t_hat[1][0], w_t_hat[1][1], w_t_hat[1][2], v_t[1]],
                            [w_t_hat[2][0], w_t_hat[2][1], w_t_hat[2][2], v_t[2]],
                            [0, 0, 0, 0]])

        u_t_adjoint = np.array([[w_t_hat[0][0], w_t_hat[0][1], w_t_hat[0][2], v_t_hat[0][0], v_t_hat[0][1], v_t_hat[0][2]],
                                [w_t_hat[1][0], w_t_hat[1][1], w_t_hat[1][2], v_t_hat[1][0], v_t_hat[1][1], v_t_hat[1][2]],
                                [w_t_hat[2][0], w_t_hat[2][1], w_t_hat[2][2], v_t_hat[2][0], v_t_hat[2][1], v_t_hat[2][2]],
                                [0, 0, 0, w_t_hat[0][0], w_t_hat[0][1], w_t_hat[0][2]],
                                [0, 0, 0, w_t_hat[1][0], w_t_hat[1][1], w_t_hat[1][2]],
                                [0, 0, 0, w_t_hat[2][0], w_t_hat[2][1], w_t_hat[2][2]]])

        ## EKF Prediction Step (lecture 13, slide 15)

        tau = t[0, i] - t[0, i-1]  # time between current and previous detection
        gaussian_noise = np.diag(np.random.normal(0,1,6))
        W = tau * tau * gaussian_noise

        mu_t_predict = np.dot(expm(-tau * u_t_hat), mu_imu)
        sigma_t_predict = np.dot(np.dot(expm(-tau * u_t_adjoint), sigma_imu), np.transpose(expm(-tau * u_t_adjoint))) + W

        ### d) IMU Update Step based on stereo camera observation

        # update imu trajectory
        trajectory_imu[:, :, i] = inv(mu_t_predict)

        # update imu mean and variance
        mu_imu = mu_t_predict
        sigma_imu = sigma_t_predict

        # update transforms
        transform_world2camera = np.dot(transform_imu2camera, mu_imu)
        transform_camera2world = inv(transform_world2camera)

        ### c) Landmark Mapping via EKF Update

        # features at this landmark
        features_current = landmarks[:, :, i]

        # sum features
        sum_of_feature_vectors = np.sum(features_current[:, :], 0)

        # if not [-1 -1 -1 -1], feature is observable
        features_obs_indices = np.array(np.where(sum_of_feature_vectors != -4))
        features_obs_indices_count = np.size(features_obs_indices)

        # feature and feature index storage variables
        features_update = np.zeros((4, 0))
        features_update_indices = np.zeros((0, 0), dtype=np.int8)

        ## Find observable features
        if features_obs_indices_count > 0:

            # observable features coordinates
            features_obs_coords = features_current[:, features_obs_indices].reshape(4, features_obs_indices_count)

            # observable features
            features_obs = np.ones((4, np.shape(features_obs_coords)[1]))
            features_obs[0, :] = (features_obs_coords[0, :] - c_u) * b / (features_obs_coords[0, :] - features_obs_coords[2, :])
            features_obs[1, :] = (features_obs_coords[1, :] - c_v) * (-M[2, 3]) / (M[1, 1] * (features_obs_coords[0, :] - features_obs_coords[2, :]))
            features_obs[2, :] = -(-f_su * b) / (features_obs_coords[0, :] - features_obs_coords[2, :])
            features_obs = np.dot(transform_camera2world, features_obs)

            ## Estimate landmark coordinates of observable features
            for j in range(features_obs_indices_count):

                m_j = features_obs_indices[0, j]  # index

                # print('m_j:',np.shape(m_j))
                # print('mu_landmarks:',np.shape(mu_landmarks))

                # if first time feature observed, add it to landmark means
                if np.array_equal(mu_landmarks[:, m_j], [-1, -1, -1, -1]):
                    mu_landmarks[:, m_j] = features_obs[:, j]
                # else append observable feature to new features array
                else:
                    features_update = np.hstack((features_update, features_obs[:, j].reshape(4, 1)))
                    features_update_indices = np.append(features_update_indices, m_j)

            features_update_count = np.shape(features_update_indices)[0]

            # check for new observable features
            if features_update_count != 0:

                # prior mean mu(t+1|t), reshape landmark matrix variable to needed form
                mu_landmark_dim = (4, features_update_count)
                mu_landmark = mu_landmarks[:, features_update_indices]
                mu_landmark.reshape(mu_landmark_dim)

                ## Jacobian H(t+1,i)

                feature_count_total = landmark_count

                # projection matrix (lecture 13, slide 7)
                P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

                for k in range(features_update_count):

                    # projection function pi(q) and its derivative d(pi)/dq (lecture 13, slide 5)
                    pi = np.dot(transform_world2camera, mu_landmark[:, k])
                    q1 = pi[0]
                    q2 = pi[1]
                    q3 = pi[2]
                    q4 = pi[3]

                    dpi_dq = (1/q3) * (np.array([[1, 0, -q1/q3, 0],
                                                 [0, 1, -q2/q3, 0],
                                                 [0, 0, 0, 0],
                                                 [0, 0, -q4/q3, 1]]))

                    ## observation model Jacobian H(t+1,i,j) evaluated at mu(t) (lecture 13, slide 8)
                    H_update = np.zeros((4 * features_update_count, 3 * feature_count_total))
                    H_update[4*k:4*(k+1), 3*m_j:3*(m_j+1)] = np.dot(np.dot(np.dot(M, dpi_dq), transform_world2camera), np.transpose(P))

                # EKF Update: Kalman gain K(t+1) (lecture 13, slide 8)
                temp_inverse = inv(np.dot(np.dot(H_update, sigma_landmarks), np.transpose(H_update)) + np.identity(4 * features_update_count) * V)
                K_update = np.dot(np.dot(sigma_landmarks, np.transpose(H_update)), temp_inverse)

                # landmark coordinates in world frame (lecture 13, slide 5)
                q = np.dot(transform_world2camera, mu_landmark)
                projection = q / q[2,:]
                z_hat = np.dot(M, projection)

                # EKF Update: landmark mean, covariance (lecture 13, slide 8)
                z = features_current[:, features_update_indices].reshape((4, features_update_count))
                mu_landmarks = (mu_landmarks.reshape(-1, 1, order='F') + np.dot(np.dot(initialization, K_update), (z - z_hat).reshape(-1, 1, order='F'))).reshape(4, -1, order='F')
                sigma_landmarks = np.dot((np.identity(3 * np.shape(landmarks)[1]) - np.dot(K_update, H_update)), sigma_landmarks)

    ### d) visualize vehicle trajectory + landmarks using function from utils.py
    fig, ax = visualize_trajectory_2d(trajectory_imu, mu_landmarks, show_ori=True)

