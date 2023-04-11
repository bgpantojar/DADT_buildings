from projection_tools import line_adjustor
import numpy as np
from ransac import ransac, LinearLeastSquaresModel
import  scipy
from scipy.spatial import distance
import cv2
import matplotlib.pyplot as plt

def fit_plane_on_X(X, fit_ransac=True):
    """
    Given a set of 3D points X it computes
    the parameters of a plane fitting them
    using least squares and ransac 
    X contains x,y,z coordinates
    the plane equation is: Ax + By + Cz + D = 0
    To use least squares: z = -(A/C)x - (B/C)y - (D/C)
    Then the data will be composed by z and params [x|y|1]
    The solution will give -A/C, -B/C, -D/C.
    The normal direction then will be (A/C,B/C,1) after normalizing it
    Args:
        X (_type_): _description_
    """

    A = np.hstack((X[:,:2], np.ones((len(X),1))))
    z = X[:,2].reshape((-1,1))
    # setup model

    all_data = np.hstack((A,z))
    input_columns = range(3) # the first columns of the array
    output_columns = [3+i for i in range(1)] # the last columns of the array
    debug = False
    model = LinearLeastSquaresModel(input_columns,output_columns,debug=debug)

    if not fit_ransac:
        fit, resids, rank, s = scipy.linalg.lstsq(all_data[:,input_columns],all_data[:,output_columns])
    else:
        # run RANSAC algorithm
        n = 10
        k = 1000
        t = .1*np.mean(distance.cdist(X,X)) #7e3
        d = .5*len(X)#100
        fit, ransac_data = ransac(all_data,model,
                                     n, k, t, d, # misc. parameters
                                     debug=debug,return_all=True)
    
    #Assuming that C is 1 in the plane equation, the normal vector is given by
    plane_normal = np.array([-fit[0][0], -fit[1][0], 1])
    plane_normal = plane_normal/np.linalg.norm(plane_normal)
    plane_params = np.array([-fit[0][0], -fit[1][0], 1, -fit[2][0]])

    return plane_normal, plane_params

def less_first(a, b):
    return [a,b] if a < b else [b,a]

def trimesh2edges(mesh):

    list_of_edges = []

    for triangle in mesh.triangles:
        for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
            list_of_edges.append(less_first(triangle[e1],triangle[e2])) # always lesser index first

    list_of_edges = np.array(list_of_edges)
    array_of_edges, counts = np.unique(list_of_edges, axis=0, return_counts=True) # remove duplicates
    array_contour_edges = list_of_edges[np.where(counts==1)] #Get edges that belong to contours

    list_of_lengths = []

    for p1,p2 in array_of_edges:
        x1 = np.array(mesh.vertices)[p1]
        x2 = np.array(mesh.vertices)[p2]
        list_of_lengths.append(np.linalg.norm(x2-x1))

    array_of_lengths = np.sqrt(np.array(list_of_lengths))

    return array_of_edges, array_of_lengths, array_contour_edges

def read_LOD(LOD_path):
    #Loading LOD model vertices and faces
    vertices = []
    triangles = []
    f = open(LOD_path, "r")
    for iii in f:
        l = iii.split()
        if l[0] == 'v':
            vertices.append([float(j) for j in l[1:]])
        else:
            for jjj in range(len(l[1:])-2):
                triangles.append([l[1], l[1+jjj+1], l[1+jjj+2]])
    f.close()
    vertices = np.array(vertices)
    triangles = np.array(triangles).astype('int')-1

    return vertices, triangles

def cluster_triangles2plane(LOD2):
    triangle_normals_round = np.round(np.array(LOD2.mesh.triangle_normals), 4) 
    unique_normals, LOD2_triangle_clusters = np.unique(triangle_normals_round, axis = 0, return_inverse=True)
    LOD2_cluster_n_triangles = [len(LOD2_triangle_clusters[np.where(LOD2_triangle_clusters==i)]) for i in range(np.max(LOD2_triangle_clusters)+1)]
    LOD2_cluster_area = np.zeros(len(LOD2_cluster_n_triangles))

    return LOD2_triangle_clusters, LOD2_cluster_n_triangles, LOD2_cluster_area

#Clossest distance of point to plane
def dist_pt2plane(plane,pt):
    dist = np.sum(plane[:3].reshape((1,3)) * pt.reshape((-1,3)), axis=1) + plane[3]
    dist = dist/(np.linalg.norm(plane[:3]))

    return dist

#Project pt to plane
def proj_pt2plane(plane,pt):
    dist = dist_pt2plane(plane,pt).reshape((-1,1))
    un = plane[:3]/(np.linalg.norm(plane[:3]))
    un = un.reshape((1,3))
    dist_matrix = np.concatenate((dist,dist,dist), axis=1)
    pt_proj = pt - dist_matrix*un
    #pt_proj = [p - d*un for p in pt for d in dist[0]]
    
    return pt_proj


def proj_pt2pl(model,pt):
     '''
     Projectin pts to plane 

     Parameters
     ----------
     model : plane ransac model
     pt : point to be projected to plane

     Returns
     -------
     pt_p : pt projection on plane
     
     model -> x: [a b c]
           -> a + bx + cy + dz = 0  (here d=-1)
           n_p: normal to plane (b,c,d)
     '''
     pt1 = np.copy(pt)
     pt1[2] = model[0] + model[1]*pt[0] + model[2]*pt[1] #point on plane
     pt_pt1 = pt - pt1
     if np.linalg.norm(pt_pt1)==0:
         pt_p=np.copy(pt)
     else:
         n_1 = pt_pt1/np.linalg.norm(pt_pt1)
         n_p = np.array([model[1] ,model[2] ,-1])
         n_p = n_p/np.linalg.norm(n_p)
         cos_alpha = np.dot(n_1,n_p)
         pt_ptp = np.linalg.norm(pt_pt1)*cos_alpha
         pt_p = pt - pt_ptp*n_p
     
     return pt_p

def proj_op2plane(polyfit_path, X, dense):
    """

        Find a plane paralel to facade with a normal similar to mean of normals
        of all openings. Project corners to that plane. 
        Project 4 points of the opening to a single plane
        Calculate normals of all openings and find the closest to the 
        facade normal. Take the openings to a plane with the same normal.

    Args:
        polyfit_path (str): path to input folder with LOD2 model
        X (array): opening coorners in 3D
        dense (bool): if true, it loads LOD2 model obtained from dense pt cloud
    Returns:
        X (array): 3D opening coordinates projected to a plane
        n_pl (array): normal parameters of plane equation
        faces_normals: faces normals of LOD2 elements
        normal_op_plane: opening normal most similar to plane normal
    """
    
    #Normals of faces of the polygonal surface (polyfit)
    #Loading polyfit model vertices and faces
    vertices_p = list()
    faces_p = list()
    if dense:
        f = open(polyfit_path +"/polyfit_dense.obj", "r")
    else:
        f = open(polyfit_path +"/polyfit.obj", "r")
    for iii in f:
        l = iii.split()
        if l[0] == 'v':
            vertices_p.append([float(j) for j in l[1:]])
        else:
            faces_p.append([int(j) for j in l[1:]])
    f.close()

    #Getting normals of faces in polyfit model
    faces_normals = list([])
    for _, f in enumerate(faces_p):
        f_vert = list([])
        for j in f:
            f_vert.append(vertices_p[j-1])
        f_vert.append(vertices_p[f[0]-1])
        f_vert = np.array(f_vert)
        v1 = f_vert[1]-f_vert[0]
        v2 = f_vert[2]-f_vert[0]
        A,B,C = np.cross(v1,v2)
        #have to be normalized to make sense the distances
        faces_normals.append((np.array([A,B,C]))/(np.linalg.norm(np.array([A,B,C]))))
    faces_normals = np.array(faces_normals)    
    
    c_v = 0 #vertices counter. Helper to asociate vertices to a single opening
    op_normals = list([]) #list with the opening normals of the facade
    #Computing opening normal directions
    for j in range(int(len(X.T)/4)):
        a = np.copy(X[0:3,c_v])
        b = np.copy(X[0:3,c_v+1])
        c = np.copy(X[0:3,c_v+2])
        d = np.copy(X[0:3,c_v+3])
        #
        #to warranty same normal direction
        #a--->v1    b
        #|         /|
        #|      v4/ |v3
        #~v2     ~  ~
        # 
        #
        #c          d 
        #
        v1 = b-a
        v2 = c-a
        A1,B1,C1 = np.cross(v1,v2)
        v3 = d-b
        v4 = c-b
        A2,B2,C2 = np.cross(v3,v4)
        A = (A1+A2)/2
        B = (B1+B2)/2
        C = (C1+C2)/2
        c_v+=4
        op_normals.append((np.array([A,B,C]))/(np.linalg.norm(np.array([A,B,C]))))
       
    op_normals = np.array(op_normals)
    mean_op_normals = np.mean(op_normals, axis=0)
    
    #Look for the normal of the faces with minimum angle with the openings
    angle_normals = list([])
    for j in faces_normals:
        angle = np.arccos(np.dot(j,mean_op_normals))*180/np.pi
        if angle > 180:
            angle -= 180
        angle_normals.append(angle)
    angle_normals = np.array(angle_normals)
           
    #conditional to be invariant to the direction of the normal vector
    if np.min(np.abs(angle_normals-180)) > np.min(np.abs(angle_normals)):
        index_normal = np.argmin(np.abs(angle_normals))
    else:
        index_normal = np.argmin(np.abs(angle_normals-180))
        
    normal_op_plane = faces_normals[index_normal]   
    
    A = normal_op_plane[0]
    B = normal_op_plane[1]
    C = normal_op_plane[2]
    n_pl = [A,B,C] 
    
    #Using perpendicular projection
    #A(x-x0) + B(y-y0) + C(z-z0) = 0 => z = (Ax0/C + By0/C + z0) - Ax/C  - By/C 
    a = X.T[0,0:3] #create a plane that pass for this point. 
    m1 = A*a[0]/C + B*a[1]/C + a[2]
    m2 = -A/C
    m3 = -B/C
    model = np.array([m1,m2,m3])
    
    for ii, XX in enumerate(X.T):
        X[:3,ii] = proj_pt2pl(model,X[:3,ii])

    return X, n_pl, faces_normals, normal_op_plane


def open2local(X, faces_normals, normal_op_plane):
    """

        #Taking corners X to a local plane.
        #Finds a local plane to project X based in the direction of the openings edges.

    Args:
        X (array): 3D opening corners coordinates
        faces_normals (array): facade elements normals
        normal_op_plane (array): normal of opening similar to the face opening
    Returns:

        X_l (array): local coordinates of 3D opening corners
        T (array): transformation matrix to map 3D opening corners from global to local

    """
    
    dir_vect_h = np.zeros((int(len(X[0,:])/2), 3))
    for ee in range(int(len(X[0,:])/4)):
        ed_h1 = (X[:3, 4*ee + 1] - X[:3, 4*ee])/np.linalg.norm((X[:3, 4*ee + 1] - X[:3, 4*ee]))
        ed_h2 = (X[:3, 4*ee + 3] - X[:3, 4*ee + 2])/np.linalg.norm(X[:3, 4*ee + 3] - X[:3, 4*ee + 2])
        dir_vect_h[2*ee] = ed_h1
        dir_vect_h[2*ee+1] = ed_h2
        
    #Choosing building normal with similar direction to the edges
    mean_dir_h = (np.mean(dir_vect_h, axis=0))/(np.linalg.norm(np.mean(dir_vect_h, axis=0)))
    #Look for the dir_vect_h with minimum angle with building normals
    ang_dir_h = list([])
    for j in faces_normals:
        angle = np.arccos(np.dot(j/np.linalg.norm(j), mean_dir_h))*180/np.pi #!new
        if angle > 180:
            angle -= 180
        ang_dir_h.append(angle)
    ang_dir_h = np.array(ang_dir_h)
           
    #conditional to be invariant to the direction of the normal vector
    if np.min(np.abs(ang_dir_h-180)) > np.min(np.abs(ang_dir_h)):
        ind_dir_h = np.argmin(np.abs(ang_dir_h))
    else:
        ind_dir_h = np.argmin(np.abs(ang_dir_h-180))
        
    normal_dir_h_plane = faces_normals[ind_dir_h] / np.linalg.norm(faces_normals[ind_dir_h]) #!new
    normal_op_plane = normal_op_plane/np.linalg.norm(normal_op_plane) #!new
    proj_norm_dir_h = normal_dir_h_plane - ((np.dot(normal_dir_h_plane,normal_op_plane))/((np.linalg.norm(normal_op_plane))**2))*normal_op_plane
    proj_norm_dir_h = proj_norm_dir_h/(np.linalg.norm(proj_norm_dir_h))
    
    A = np.copy(X[:3,0])
    B = A + proj_norm_dir_h
    N = np.copy(normal_op_plane)
    U = (B - A)/np.linalg.norm(B-A)
    V = np.cross(N,U)
    u = A + U
    v = A + V
    n = A + N
    #Solving the sistem for T:
    G = np.ones((4,4))
    G[:3,0] = np.copy(A)
    G[:3,1] = np.copy(u)
    G[:3,2] = np.copy(v)
    G[:3,3] = np.copy(n)
    
    L = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,1,1]])
    
    T = np.dot(L,np.linalg.inv(G))
    
    #Finding local coordinates Xl with T matrix
    Xl = np.dot(T,X)
    Xl = np.round(Xl,6)

    return Xl, T


def op_aligning1(Xl, cte = .05):
    """
    Aligning the width and height of the openings (Aligment 1 --> to linear regression model).

    Args:
        Xl (array): local coordinates of opening corners
    Returns:
        Xl_al (array): aligned local coordinates of opening corners
    """
    #print("THE CONSTANT IS ", cte)    
    print("Aligning 1---")    
    #Threshold (depends on how the keypoints are organized)#!
    threshold = cte*np.min(np.abs(((Xl[0,0]-Xl[0,1]),(Xl[1,0]-Xl[1,2]))))
    #threshold = cte*np.abs(np.min(((Xl[0,0]-Xl[0,1]),(Xl[1,0]-Xl[1,2]))))
    
    #ALIGNING TO LINES
    #Vertical alignment
    vert_checker = np.zeros(Xl[0].shape) #Checker to identify points already aligned
    Xl_al = np.copy(Xl) #if there is problems with not meeting the threshold, keep initial
    for ii, pt in enumerate(Xl[0,:].T):
        if vert_checker[ii] == 0:
            distances = np.abs(pt - Xl[0]) #!make linear regression + RANSAC instead i think
            meet_thr = np.where(distances<threshold)
            if ii%2==0: #if are in the left corners
                left_meet_thr = np.where(meet_thr[0]%2==0)
                meet_thr = meet_thr[0][left_meet_thr]
            else: #right
                right_meet_thr = np.where(meet_thr[0]%2!=0)
                meet_thr = meet_thr[0][right_meet_thr]
            if np.sum(vert_checker[meet_thr])==0: #to avoid take again points already aligned    
                x_lr = Xl[0][meet_thr]
                y_lr = Xl[1][meet_thr]
                if len(x_lr)>2: #guarantee to do regretion 
                    Xl_al[0][meet_thr], Xl_al[1][meet_thr] = line_adjustor(x_lr,y_lr)
                else: #If just two poits, they are from same openints. Put the same x coordinate
                    Xl_al[0][meet_thr] = np.mean(Xl_al[0][meet_thr])
            vert_checker[meet_thr] = 1
    
    #Horizontal alignment
    hori_checker = np.zeros(Xl[1].shape) #Checker to identify points already aligned
    for ii, pt in enumerate(Xl_al[1,:]):
        if hori_checker[ii] == 0:
            distances = np.abs(pt - Xl_al[1])
            meet_thr = np.where(distances<threshold)
            if ii%4==0 or ii%4==1: #top corners
                top_meet_thr1 = np.where(meet_thr[0]%4==0)
                top_meet_thr2 = np.where(meet_thr[0]%4==1)
                top_meet_thr = np.concatenate((top_meet_thr1,top_meet_thr2),axis=1)
                meet_thr = meet_thr[0][top_meet_thr]
            else: #bottom
                bottom_meet_thr1 = np.where(meet_thr[0]%4==2)
                bottom_meet_thr2 = np.where(meet_thr[0]%4==3)
                bottom_meet_thr = np.concatenate((bottom_meet_thr1,bottom_meet_thr2),axis=1)
                meet_thr = meet_thr[0][bottom_meet_thr]
            if np.sum(hori_checker[meet_thr])==0: #to avoid take again points already aligned
                x_lr = Xl_al[0][meet_thr][0]
                y_lr = Xl_al[1][meet_thr][0]
                if len(x_lr)>2:
                    Xl_al[0][meet_thr], Xl_al[1][meet_thr] = line_adjustor(x_lr,y_lr)
                else: #If just two poits, they are from same openints. Put the same x coordinate
                    Xl_al[1][meet_thr] = np.mean(Xl_al[1][meet_thr])
            hori_checker[meet_thr] = 1

    return Xl_al


def op_aligning2(Xl_al, cte = .35):
    """

    Aligning the width and height of the openings (Aligment 2 --> same width and height)

    Args:
        Xl_al (array): aligned local coordinates of opening corners (linear regression)
    Return:
        Xl_al2 (array): aligned local coordinates of opening corners (same width - height)
    """
    print("Aligning 2---")       
    #Same width and height for openings#!    
    threshold = cte*(np.min((np.abs(Xl_al[0,0]-Xl_al[0,1]),np.abs(Xl_al[1,0]-Xl_al[1,2]))))     
       
    #Vertical alignment
    vert_checker = np.zeros(Xl_al[0].shape) #Checker to identify points already aligned
    Xl_al2 = np.copy(Xl_al) #if there is problems with not meeting the threshold, keep initial
    for ii, pt in enumerate(Xl_al[0,:]):
        if vert_checker[ii] == 0:
            distances = np.abs(pt - Xl_al[0])
            meet_thr = np.where(distances<threshold)
            if ii%2==0: #if are in the left corners
                left_meet_thr = np.where(meet_thr[0]%2==0)
                meet_thr = meet_thr[0][left_meet_thr]
            else: #right
                right_meet_thr = np.where(meet_thr[0]%2!=0)
                meet_thr = meet_thr[0][right_meet_thr]
            if np.sum(vert_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_coordinate = np.mean(Xl_al[0][meet_thr])
                Xl_al2[0][meet_thr] = np.copy(mean_coordinate)
            vert_checker[meet_thr] = 1
    #Horizontal alignment
    hori_checker = np.zeros(Xl_al[0].shape) #Checker to identify points already aligned
    for ii, pt in enumerate(Xl_al[1,:]):
        if hori_checker[ii] == 0:
            distances = np.abs(pt - Xl_al[1])
            meet_thr = np.where(distances<threshold)
            if ii%4==0 or ii%4==1: #top corners
                top_meet_thr1 = np.where(meet_thr[0]%4==0)
                top_meet_thr2 = np.where(meet_thr[0]%4==1)
                top_meet_thr = np.concatenate((top_meet_thr1,top_meet_thr2), axis=1)
                meet_thr = meet_thr[0][top_meet_thr]
            else: #bottom
                bottom_meet_thr1 = np.where(meet_thr[0]%4==2)
                bottom_meet_thr2 = np.where(meet_thr[0]%4==3)
                bottom_meet_thr = np.concatenate((bottom_meet_thr1,bottom_meet_thr2), axis=1)
                meet_thr = meet_thr[0][bottom_meet_thr]
            if np.sum(hori_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_coordinate = np.mean(Xl_al[1][meet_thr])
                Xl_al2[1][meet_thr] = np.copy(mean_coordinate)
            hori_checker[meet_thr] = 1
    

    return Xl_al2


def op_aligning3(Xl_al2, cte1 = .1, cte2 = .3):
    """

    Equalizing areas. Aligning cetroids. Calculating area of each opening. Increment or decrease
    edges to have same area.

    Args:
        Xl_al2 (array): aligned local coordinates of opening corners (same width - height)

    Returns:
        Xl_al3 (array): aligned local coordinates of opening corners (equal areas)
    """
    print("Aligning 3---")    
    #print("THE CONSTANTS ARE ", cte1, cte2)    
        
    Xl_al3 = np.copy(Xl_al2)
    centroids = [] 
    areas = []
    edges_h = []
    edges_v = []
    for j in range(int(Xl_al2.shape[1]/4)):
        xc = (Xl_al2.T[4*j,0] + Xl_al2.T[4*j+1,0])/2
        yc = (Xl_al2.T[4*j,1] + Xl_al2.T[4*j+2,1])/2
        centroids.append([xc,yc])
        edge_h = np.abs(Xl_al2.T[4*j,0] - Xl_al2.T[4*j+1,0])
        edge_v = np.abs(Xl_al2.T[4*j,1] - Xl_al2.T[4*j+2,1])
        edges_h.append(edge_h)
        edges_v.append(edge_v)
        areas.append(edge_h*edge_v)
    centroids = np.array(centroids)
    areas = np.array(areas)
    edges_h = np.array(edges_h)
    edges_v = np.array(edges_v)
    
    
    #Vertical centroids aligment#!
    threshold = cte1*np.min(np.abs(((Xl_al3[0,0]-Xl_al3[0,1]),(Xl_al3[1,0]-Xl_al3[1,2]))))     
    vert_checker = np.zeros(len(centroids)) #Checker to identify points already aligned
    centroids_al = np.copy(centroids) 
    #centroids_al = np.zeros_like(centroids) 
    for ii, pt in enumerate(centroids[:,0]):
        if vert_checker[ii] == 0:
            distances = np.abs(pt - centroids[:,0])
            meet_thr = np.where(distances<threshold)
            if len(meet_thr[0])>0 and np.sum(vert_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_coordinate = np.mean(centroids[:,0][meet_thr])
                centroids_al[meet_thr,0] = np.copy(mean_coordinate)
            vert_checker[meet_thr] = 1
    #Horizontal centroids alignment
    hori_checker = np.zeros(len(centroids)) #Checker to identify points already aligned
    for ii, pt in enumerate(centroids[:,1]):
        if hori_checker[ii] == 0:
            distances = np.abs(pt - centroids[:,1])
            meet_thr = np.where(distances<threshold)
            if len(meet_thr[0])>0 and np.sum(hori_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_coordinate = np.mean(centroids[:,1][meet_thr])
                centroids_al[meet_thr,1] = np.copy(mean_coordinate)
            hori_checker[meet_thr] = 1
    
    #Equalizing areas - to establish a threshold in the area diferences. #!
    threshold = cte2*np.min(areas)
    area_checker = np.zeros(len(areas))
    edges_h_e = np.copy(edges_h)
    edges_v_e = np.copy(edges_v)
    for ii, ar in enumerate(areas):
        if area_checker[ii] == 0:
            diferences = np.abs(ar - areas)
            meet_thr = np.where(diferences<threshold)
            if len(meet_thr[0])>0 and np.sum(area_checker[meet_thr])==0: #to avoid take again points already aligned
                mean_edge_h = np.mean(edges_h[meet_thr])
                mean_edge_v = np.mean(edges_v[meet_thr])
                edges_h_e[meet_thr] = np.copy(mean_edge_h)
                edges_v_e[meet_thr] = np.copy(mean_edge_v)
            area_checker[meet_thr] = 1
    
    #Generation new coordinates for openings with same area
    for j in range(int(Xl_al3.shape[1]/4)):
        #x coordinates
        Xl_al3[0,4*j]   = centroids_al[j][0] + edges_h_e[j]/2
        Xl_al3[0,4*j+1] = centroids_al[j][0] - edges_h_e[j]/2
        Xl_al3[0,4*j+2] = centroids_al[j][0] + edges_h_e[j]/2
        Xl_al3[0,4*j+3] = centroids_al[j][0] - edges_h_e[j]/2
        #y coordinates
        Xl_al3[1,4*j]   = centroids_al[j][1] - edges_v_e[j]/2
        #Xl_al3[1,4*j]   = centroids_al[j][1] + edges_v_e[j]/2
        Xl_al3[1,4*j+1] = centroids_al[j][1] - edges_v_e[j]/2
        #Xl_al3[1,4*j+1] = centroids_al[j][1] + edges_v_e[j]/2
        Xl_al3[1,4*j+2] = centroids_al[j][1] + edges_v_e[j]/2
        #Xl_al3[1,4*j+2] = centroids_al[j][1] - edges_v_e[j]/2
        Xl_al3[1,4*j+3] = centroids_al[j][1] + edges_v_e[j]/2
        #Xl_al3[1,4*j+3] = centroids_al[j][1] - edges_v_e[j]/2
        
    #Testing final areas
    f_areas = []
    for j in range(int(Xl_al3.shape[1]/4)):
        edge_h = np.abs(Xl_al3.T[4*j,0] - Xl_al3.T[4*j+1,0])
        edge_v = np.abs(Xl_al3.T[4*j,1] - Xl_al3.T[4*j+2,1])
        f_areas.append(edge_h*edge_v)
    
    return Xl_al3

def plot_tn_kinematic(crack_kinematic_path, crack_mask_path, plot_trans_n=False, plot_trans_t = False, plot_trans_t_n=False, resolution=None, dot_size=None):

    lt="_loc"
    
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    #Creating colormap
    
    top = cm.get_cmap('Oranges_r', 256)
    bottom = cm.get_cmap('Blues', 256)
    newcolors = np.vstack((top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256))))
    newcmp = ListedColormap(newcolors, name='OrangeBlue')    
    

    #Reading mask        
    mask = cv2.imread(crack_mask_path)
    mask = (mask==0)*255
    
    #Reading skeleton and rotation information
    glob_coordinates_skl = []
    two_dofs_n = []
    two_dofs_t = []
    crack_class = []

    import json
    
    # Opening JSON file
    with open(crack_kinematic_path) as json_file:
        crack_kinematic = json.load(json_file)

    for lab in crack_kinematic:
        glob_coordinates_skl = glob_coordinates_skl + [coord[0] for coord in crack_kinematic[lab]["kinematics_n_t"+lt]]
        two_dofs_n = two_dofs_n + [t_dofs_n[2][0] for t_dofs_n in crack_kinematic[lab]["kinematics_n_t"+lt]]
        two_dofs_t = two_dofs_t + [t_dofs_t[2][1] for t_dofs_t in crack_kinematic[lab]["kinematics_n_t"+lt]]
        crack_class = crack_class + [cr_cl for cr_cl in crack_kinematic[lab]["crack_class"]]
    
    glob_coordinates_skl = np.array(glob_coordinates_skl, 'int')
    two_dofs_n = np.array(two_dofs_n)
    two_dofs_t = np.array(two_dofs_t)
    crack_class = np.array(crack_class)
    
    #Modifing signs according new sign sistem. Opening is possitive (always). Shear sliding, clockwise pair positive.
    #If class 1 (crack ascending), it is necessary to change sings with respect old convention
    #If class 2 (crack descending), it is necessary to keep sings with respect old convention
    #if sign_convention=="new": #check later -- it might influence in the errory displa
    two_dofs_n = np.abs(two_dofs_n)
    ind_class1 = np.where(crack_class==1)
    two_dofs_t[ind_class1[0]] *= -1

    if plot_trans_n:
        trans_n_img = np.zeros_like(mask, 'float')
        trans_n_img = trans_n_img[:,:,0]
        trans_n_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_n
        
        if resolution is None:
            a = 1
        else:
            a = resolution ##mm/px
        
        #Ploting 
        fig, ax = plt.subplots(1)        
            
        if dot_size is not None:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*two_dofs_n, cmap=bottom, vmin=0 , vmax=np.max(a*two_dofs_n), marker='.', s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*two_dofs_n, cmap=bottom, vmin=0 , vmax=np.max(a*two_dofs_n), marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        
        if resolution is None:
            clrbr.ax.set_title(r"$t_n [px]$")
        else:
            clrbr.ax.set_title(r"$t_n [mm]$")
        
        fig.savefig('../results/n_t_kin_tn'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig('../results/n_t_kin_tn'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if plot_trans_t:
        trans_t_img = np.zeros_like(mask, 'float')
        trans_t_img = trans_t_img[:,:,0]
        trans_t_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_t
        
        if resolution is None:
            a = 1
        else:
            a = resolution ##mm/px
        
        #Ploting 
        fig, ax = plt.subplots(1)        
        #TO MAKE IT COMPARABLE WITH DIC METHOD, THE SIGN NEED TO BE CHANGED
        c_ = two_dofs_t
        if dot_size is not None:
            #psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*c_, cmap=newcmp, vmin=-20 , vmax=+20, marker='.', s=dot_size)
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*c_, cmap=newcmp, vmin=np.min(a*c_) , vmax=np.max(a*c_), marker='.', s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=a*c_, cmap=newcmp, vmin=np.min(a*c_) , vmax=np.max(a*c_), marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        if resolution is None:
            clrbr.ax.set_title(r"$t_t [px]$")
        else:
            clrbr.ax.set_title(r"$t_t [mm]$")
        
        if resolution is not None: lt+="_mm"
        fig.savefig('../results/n_t_kin_tt'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig('../results/n_t_kin_tt'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        
    if plot_trans_t_n:
        # img
        trans_t_n_img = np.zeros_like(mask, 'float')
        trans_t_n_img = trans_t_n_img[:,:,0]
        trans_t_n_img[(glob_coordinates_skl[:,1], glob_coordinates_skl[:,0])] = two_dofs_t/two_dofs_n
        
        #Ploting 
        fig, ax = plt.subplots(1)        
        
        #TO MAKE IT COMPARABLE WITH DIC METHOD, THE SIGN NEED TO BE CHANGED
    
        c_ = two_dofs_t/two_dofs_n

        if dot_size is not None:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=c_, cmap=newcmp, vmin=-2 , vmax=2, marker='.', s=dot_size)
        else:
            psm = ax.scatter(glob_coordinates_skl[:,0], glob_coordinates_skl[:,1], c=c_, cmap=newcmp, vmin=-2, vmax=2, marker='.')
        ax.imshow(mask, alpha=0.3)
        clrbr = fig.colorbar(psm, ax=ax)
        clrbr.ax.set_title(r"$t_t/t_n$")
        
        fig.savefig('../results/n_t_kin_tt_tn'+lt+'.png', bbox_inches='tight', pad_inches=0)
        fig.savefig('../results/n_t_kin_tt_tn'+lt+'.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

def rect_overlap(R1, R2):
      if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
         return False
      else:
         return True
