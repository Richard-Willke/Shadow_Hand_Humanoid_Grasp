import time
import mujoco
import mujoco.viewer
import os.path
import platform
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from IPython.display import clear_output
clear_output()

#insert path to your project here
directory = os.path.dirname(os.path.abspath(__file__))
hand = "shadow_hand"
left_or_right = "left"
if left_or_right[0] == 'l' or left_or_right[0] == 'L':
    left_or_right = "scene_left_torque_1.xml"
else:
    left_or_right = "right_hand.xml"

if platform.system() == "Linux":

     m = mujoco.MjModel.from_xml_path(directory + '/'+hand+'/'+left_or_right)

elif platform.system() == "Windows":

    m = mujoco.MjModel.from_xml_path(directory + '\\'+hand+'\\'+left_or_right)

else:
    print("We don't respect Mac'")


d = mujoco.MjData(m)

#model info
nDOF = m.nv

#define Finger Tips
fingerTips = [m.body('lh_thdistal').id,m.body('lh_ffdistal').id,m.body('lh_mfdistal').id,m.body('lh_rfdistal').id,m.body('lh_lfdistal').id]
#actuatedJoints = [m.jnt('lh_FFJ0').id,m.jnt('lh_FFJ3').id,m.jnt('lh_FFJ4').id,m.jnt('lh_LFJ0').id,m.jnt('lh_LFJ3').id,m.jnt('lh_LFJ4').id,m.jnt('lh_LFJ5').id,m.jnt('lh_MFJ0').id,
                 # m.jnt('lh_MFJ3').id,m.jnt('lh_MFJ4').id,m.jnt('lh_RFJ0').id,m.jnt('lh_RFJ3').id,m.jnt('lh_RFJ4').id,m.jnt('lh_THJ1').id,
                  #m.jnt('lh_THJ2').id,m.jnt('lh_THJ3').id,m.jnt('lh_THJ4').id,m.jnt('lh_THJ5').id,m.jnt('lh_WRJ1').id,m.jnt('lh_WRJ2').id]


def quat2SO3(R,quat):
    # was to dumb to use the mujoco.mju_quat2Mat() implementation
    # however this should work to just a bit less efficient
    R[0,0] = 1-2*(quat[2]*quat[2]+quat[3]*quat[3]) 
    R[1,1] = 1-2*(quat[1]*quat[1]+quat[3]*quat[3]) 
    R[2,2] = 1-2*(quat[1]*quat[1]+quat[2]*quat[2]) 
    R[0,1] = 2*(quat[1]*quat[2]-quat[0]*quat[3])
    R[0,2] = 2*(quat[1]*quat[3]+quat[0]*quat[2])
    R[1,0] = 2*(quat[1]*quat[2]+quat[0]*quat[3])
    R[1,2] = 2*(quat[2]*quat[3]-quat[0]*quat[1])
    R[2,0] = 2*(quat[1]*quat[3]-quat[0]*quat[2])
    R[2,1] = 2*(quat[2]*quat[3]+quat[0]*quat[1])

## Coupling Matrix
A = np.zeros([20,30])
A[0,0] = 1
A[1,1] = 1
A[2,19] = 1
A[3,20] = 1
A[4,21] = 1
A[5,22] = 1
A[6,23] = 1
A[7,2] = 1
A[8,3] = 1
A[9,4] = 1
A[9,5] = 1
A[10,6] = 1
A[11,7] = 1
A[12,8] = 1
A[12,9] = 1
A[13,10] = 1
A[14,11] = 1
A[15,12] = 1
A[15,13] = 1
A[16,14] = 1
A[17,15] = 1
A[18,16] = 1
A[19,17] = 1
A[19,18] = 1


## Input

# reset Data
mujoco.mj_resetData(m, d)

# initial State: Objecet --> spin it realy fast
d.qvel[27:30] =1*np.random.randn(3)

#controller
P_trans = np.array([[100,0,0],[0,100,0],[0,0,100]])
D_trans = np.array([[5,0,0],[0,5,0],[0,0,5]]) #TODO: If time implement damping design

P_rot = np.array([[0,0,0],[0,0,0],[0,0,0]])
D_rot = np.array([[0,0,0],[0,0,0],[0,0,0]]) #TODO: If time implement damping design

def impedanceModel(x_des,P_trans,D_trans,P_rot,D_rot,m,d,J_t,J_r,R_tip,x_err,vel,fingerTips,A):
    x_err = x_des-d.xpos[fingerTips]

    f_des = np.zeros([5,3])
    tau_des = np.zeros(d.ctrl.shape)
    
    for i in range(0,5):
        vel[i] = np.einsum('ij,j',J_t[i],d.qvel)
        #f_des[i,:] = np.einsum('ij,j',R_tip[i]*P_trans*R_tip[i].transpose(),x_err[i,:])-np.einsum('ij,j',R_tip[i]*D_trans*R_tip[i].transpose(),vel[i])
        f_des[i,:] = np.einsum('ij,j',P_trans,x_err[i,:])-np.einsum('ij,j',D_trans.transpose(),vel[i])
        tau_i = np.einsum('ij,j',J_t[i].transpose(),f_des[i,:])
        tau_des = tau_des + np.einsum('ij,j',A,tau_i)

    ##TODO: angles and angular velocities 
    ##TODO: map Wrenches back onto joint space

    return tau_des
    
    
# Actuator signal
#d.ctrl[18] =1.5

d.actuator('lh_A_FFJ0')


# visualize contact frames and forces, make body transparent
options = mujoco.MjvOption()
mujoco.mjv_defaultOption(options)
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False # Object is  set to transparent in rgb value in XML

# tweak scales of contact visualization elements
m.vis.scale.contactwidth = 0.3
m.vis.scale.contactheight = 0.02
m.vis.scale.forcewidth = 0.05
m.vis.map.force = 0.03


#Simulation Settings
n_steps = 400
height = 240
width = 320
frames = []
fps = 60
#WarmUp_Time = 0.2 # stepping into the simulation a couple times so the R matrices don't get initialized with NONE

#Initializations
sim_time = np.zeros(n_steps)
COM_recorded = np.zeros([n_steps,3])
x_smallFinger = np.zeros([n_steps,3])
x_ringFinger = np.zeros([n_steps,3])
x_middleFinger = np.zeros([n_steps,3])
x_foreFinger = np.zeros([n_steps,3])
#see_quats = np.zeros([n_steps,4])
x_thumb = np.zeros([n_steps,3])
x_des = np.zeros([5,3])+d.xpos[m.body('object').id]
jac_t = [np.zeros([3, nDOF]),np.zeros([3, nDOF]),np.zeros([3, nDOF]),np.zeros([3, nDOF]),np.zeros([3, nDOF])] # initialize translation jacobian
jac_r = [np.zeros([3, nDOF]),np.zeros([3, nDOF]),np.zeros([3, nDOF]),np.zeros([3, nDOF]),np.zeros([3, nDOF])] # initialize rotational jacobian
R_Tip = [np.zeros([3,3]),np.zeros([3,3]),np.zeros([3,3]),np.zeros([3,3]),np.zeros([3,3])]
goal = [0.49, 0.13, 0.59]
x_err = np.zeros([n_steps,3])
#x_err = np.zeros([5,3])
vel = [np.einsum('ij,j',jac_t[0],d.qvel),np.einsum('ij,j',jac_t[1],d.qvel),np.einsum('ij,j',jac_t[2],d.qvel),np.einsum('ij,j',jac_t[3],d.qvel),np.einsum('ij,j',jac_t[4],d.qvel)]
# Simulate and display video.


#Perform one initialization Step 
#Step
#mujoco.mj_step(m, d)
#sim_time[0] = d.time

# Kinematics
#mujoco.mj_forward(m, d);
#for i in range(0,4):
#    mujoco.mj_jac(m,d,jac_t[i],jac_r[i],x_des[i],fingerTips[i])
#see_quats[0,:] = d.xquat[fingerTips[4]]

def desired_grasp_forces(F_g_obj, jacobian, friction_coeff, obj_pos, fingers_pos):
    r = []
    twists = []
    twist_obj = np.array([[0],
                          [0],
                          [0],
                          [0],
                          [0],
                          [F_g_obj]])
    tau = 0 #[]
    
    for i in range(len(fingers_pos)):
        r_ci = np.sqrt((fingers_pos[0] - obj_pos[0])**2 + (fingers_pos[1] - obj_pos[1])**2 + (fingers_pos[2] - obj_pos[2])**2)
        r.append(r_ci)
        #need to redo angle y!!!!!!!!!!!!!!!!!!
        angle_y = -(np.pi/2 - np.arcsin((fingers_pos[1] - obj_pos[1])/(fingers_pos[0] - obj_pos[0])))
        angle_z = ( np.arctan((fingers_pos[2] - obj_pos[2])/np.sqrt((fingers_pos[0] - obj_pos[0])**2 + (fingers_pos[1] - obj_pos[1])**2)))

        # assume x axis always points to obj center
        rot_y = np.array([[np.cos(angle_y), 0,  np.sin(angle_y)],
                          [0,                              1, 0],
                          [-np.sin(angle_y), 0, np.cos(angle_y)]])
        rot_z = np.array([[1,                              0, 0],
                          [0, np.cos(angle_z),  np.sin(angle_z)],
                          [0, -np.sin(angle_z), np.cos(angle_z)]])
        R_s_ci = rot_y @ rot_z
        R_s_ci_stack = np.vstack(np.hstack(R_s_ci, np.zeros((3,3))),np.hstack(np.zeros(3,3),R_s_ci))

        S_r_ci = r_ci * np.array([[0, -(fingers_pos[2] - obj_pos[2]), (fingers_pos[1] - obj_pos[1])],
                                 [(fingers_pos[2] - obj_pos[2]), 0, -(fingers_pos[0] - obj_pos[0])],
                                 [-(fingers_pos[1] - obj_pos[1]), (fingers_pos[0] - obj_pos[0]), 0]])
        P_i = np.vstack(np.hstack(np.eye(3), np.zeros((3,3))),np.hstack(S_r_ci,np.eye(3)))
        B = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]])
        G_i = B @ R_s_ci_stack.T @ P_i.T
        
        twist_contact_i = G_i @ twist_obj
        twists.append(twist_contact_i)
        
        rot_y_pi_half = np.array([[0,  0,  1],
                                  [0,  1,  0],
                                  [-1, 0,  0]])
        #assume, since we only turn contact coordinate system on y and z, that we just have to turn in pi/2 direction for y to get the correct force direction
        f_grasp_i_min =  friction_coeff * rot_y_pi_half @ twist_contact_i[3:]
        #f_i.append(f_grasp_i_min)
        tau_i = - jacobian @ np.array[[[0],[0],[0],[f_grasp_i_min[0]],[f_grasp_i_min[1]],[f_grasp_i_min[2]]]]
        tau += tau_i #.append(tau_i)
    return tau



with mujoco.Renderer(m, height, width) as renderer:
  for i in range(0,n_steps):
    print(i)
    while d.time < i/fps:
        #Step
        mujoco.mj_step(m, d)
        sim_time[i] = d.time

        # Kinematics
        #mujoco.mj_forward(m, d);
        for i in range(0,5):
            mujoco.mj_jac(m,d,jac_t[i],jac_r[i],x_des[i],fingerTips[i])
        quat2SO3(R_Tip[0],d.xquat[fingerTips[0]])
        quat2SO3(R_Tip[1],d.xquat[fingerTips[1]])
        quat2SO3(R_Tip[2],d.xquat[fingerTips[2]])
        quat2SO3(R_Tip[3],d.xquat[fingerTips[3]])
        quat2SO3(R_Tip[4],d.xquat[fingerTips[4]])

        # Controller
        x_des[:,0:3] = d.xpos[m.body('object').id]
        tau_des = impedanceModel(x_des,P_trans,D_trans,P_rot,D_rot,m,d,jac_t,jac_r,R_Tip,x_err,vel,fingerTips,A)
        
        #tau_des_grasp = desired_grasp_forces(F_g_obj = None, jacobian = None, friction_coeff = None, obj_pos = None, fingers_pos = None) #replace none with actual values 

        d.ctrl = tau_des
        # Data for plotting
        COM_recorded[i,:] = d.xpos[m.body('object').id]
        x_smallFinger[i,:] = d.xpos[fingerTips[4]]
        x_ringFinger[i,:] = d.xpos[fingerTips[3]]
        x_middleFinger[i,:]= d.xpos[fingerTips[2]]
        x_foreFinger[i,:] = d.xpos[fingerTips[1]]
        x_thumb[i,:] = d.xpos[fingerTips[0]]


        
        renderer.update_scene(d, "track",options)
        frame = renderer.render()
        frames.append(frame)
        

#media.show_video(frames, fps=fps)
import cv2
for frame in frames:
    cv2.imshow("Video", frame)
    key = cv2.waitKey(int(1000 / fps))  # Delay based on desired fps
    if key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()