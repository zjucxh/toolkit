import numpy as np
import logging 
def read_pose(data_path:str)->np.ndarray:
    pose = np.load(data_path)
    pose = pose['poses'][:,:72]
    pose[:,66:72] = 0.0 # reset hand
    #logging.debug(f' pose shape : {pose.shape}')
    return pose

def read_beta(data_path:str)->np.ndarray:
    betas = np.load(data_path)
    betas = betas['betas'][:,:10]
    return betas

def load_obj(file_obj:str):
    vertices = []
    faces = []
    with open(file_obj, 'r') as file:
        for line in  file:
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                vertex = np.array([float(values[1]), float(values[2]), float(values[3])])
                vertices.append(vertex)
            if values[0] == 'f':
                face = [int(value.split('/')[0]) - 1 for value in values[1:]]
                faces.append(face)
    vertices = np.array(vertices,dtype=np.float64)
    faces = np.array(faces,dtype=np.int32)
    return vertices, faces


def compute_deformation_gradient(A, B):
    r'''
    Compute deformation gradient for each triangle
    A: undeformed vertex coordinates (3x3 matrix)
    B: deformed vertex coordinates (3x3 matrix)

    '''
    
    # Compute the displacement vectors
    U = B - A

    # Compute the displacement gradient matrix
    D = U / A
    print(f' D : {D}')
    # Compute the deformation gradient tensor
    F = np.transpose(D)

    return F

if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]])
    B = 3 * A

    F = compute_deformation_gradient(A,B)
    
    logging.debug(f' F : {F}')
