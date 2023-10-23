import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv
import math

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        # Initialize kalman variables
        self.dt = 0.035 # time-step = 35ms
        self.G = np.array([[self.dt, 0],
                            [0,self.dt]]) #Conntrol Matrix
        self.process_cov = np.array([[0.5,0],
                                     [0,0.5]])
        self.x_prev = 0
        self.y_prev = 0
        self.pos_prev = np.array([[self.x_prev,0],
                                  [0,self.y_prev]])
        self.K = 0
        self.mes_cov = np.array([[0.01,0],
                                 [0,0.01]])
        self.yaw = 0
        self.vc = np.zeros(2)
        self.estimated = Odometry()
        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)
        self.odom = self.create_subscription(Odometry,'/odom',callback=self.Vel,qos_profile=1)
        #publish the estimated reading
        self.estimated_pub=self.create_publisher(Odometry,
                                                 "/odom_estimated",1)

    def Vel(self,msg):
        current = msg
        self.yaw = math.atan2(2.0 * (current.pose.pose.orientation.w * current.pose.pose.orientation.z + current.pose.pose.orientation.x * current.pose.pose.orientation.y), 1.0 - 2.0 * (current.pose.pose.orientation.y * current.pose.pose.orientation.y + current.pose.pose.orientation.z * current.pose.pose.orientation.z))
        self.vc = np.array([current.twist.twist.linear.x * math.cos(self.yaw), current.twist.twist.linear.x * math.sin(self.yaw)]) # Current Velocity matrix
        print(math.degrees(self.yaw))
        # print(self.vc)


    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        current = msg
        #Current x and y
        mes_c = np.array([[current.pose.pose.position.x,0],
                           [0,current.pose.pose.position.y]]) # Current position matrix
        # yaw =2 * math.acos(current.pose.pose.orientation.w) # yaw degree
        # yaw = math.atan2(2.0 * (current.pose.pose.orientation.y * current.pose.pose.orientation.z + current.pose.pose.orientation.w * current.pose.pose.orientation.x), current.pose.pose.orientation.w * current.pose.pose.orientation.w - current.pose.pose.orientation.x * current.pose.pose.orientation.x - current.pose.pose.orientation.y * current.pose.pose.orientation.y + current.pose.pose.orientation.z * current.pose.pose.orientation.z)
        # vc = np.array([current.twist.twist.linear.x * math.cos(yaw), current.twist.twist.linear.x * math.sin(yaw)]) # Current Velocity matrix
        # print(math.degrees(self.yaw))

        # Prediction step
        # Predict State:
        pos_pred = np.zeros((2,2))
        pos_pred = self.pos_prev + np.matmul(self.vc, self.G) # Prediction
        # print(self.process_cov)
        # print(self.K)
        # Predict Error Covarience:
        cov_pred = self.process_cov
        #Kalman filter step:
        self.K = np.matmul(self.process_cov, inv(self.process_cov + self.mes_cov))
        # print(self.process_cov)
        # Update step
        
        pos_n = pos_pred + np.matmul(self.K,(mes_c - pos_pred))
        
        cov_n = np.matmul((np.identity(2) - self.K),cov_pred)
        #publish the estimated reading
        self.estimated.pose.pose.position.x = pos_n[0][0]

        self.estimated.pose.pose.position.y = pos_n[1][1]

        # self.estimated.pose.pose.position.x = pos_pred[0][0]

        # self.estimated.pose.pose.position.y = pos_pred[1][1]
        
        self.process_cov = cov_n - np.matmul(self.K, cov_n)
        self.pos_prev = pos_n

        self.estimated_pub.publish(self.estimated)
        pass    

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
