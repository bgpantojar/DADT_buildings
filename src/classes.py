import os
import numpy as np
import copy
from utils_geometry import rect_overlap, fit_plane_on_X, trimesh2edges, read_LOD, cluster_triangles2plane, proj_pt2plane, proj_op2plane, open2local, op_aligning1, op_aligning2, op_aligning3
import pymeshlab
import open3d as o3d
import open3d_tutorial as o3dtut
import matplotlib.pyplot as plt



class domain:
    def __init__(self) -> None:
        self.LOD2 = None
        self.openings = []
        self.LOD3 = None
        self.cracks = []
    
    def load_LOD2(self, dense, polyfit_path):
        #Reading LOD2 obj file and creating mesh
        #Reading LOD2 model as mesh and creating a clustered point cloud out of it
        #Loading polyfit model vertices and faces
        if dense:
            polyfit_obj_path = polyfit_path + "/polyfit_dense.obj"
        else:
            polyfit_obj_path = polyfit_path + "/polyfit.obj"

        LOD2_vertices, LOD2_triangles = read_LOD(polyfit_obj_path) 
        LOD2_mesh = o3d.geometry.TriangleMesh()
        LOD2_mesh.vertices = o3d.utility.Vector3dVector(LOD2_vertices) 
        LOD2_mesh.triangles = o3d.utility.Vector3iVector(LOD2_triangles) 
        LOD2_mesh.compute_vertex_normals()
        self.LOD2 = LOD2(LOD2_mesh)
    
    def assign_openings2planes(self):
        for opening in self.openings:
            self.LOD2.planes[opening.plane].openings.append(opening)
        
    def detect_redundant_openings(self, parameter = "size"):
        for plane in self.LOD2.planes:
            if len(plane.openings)==0:
                continue
            X_openings = np.empty(shape=(0,3))
            #Read openings coordinates
            for opening in plane.openings:
                X_openings = np.concatenate((X_openings, opening.coord), axis=0)            
            #Project all opening's corners to the plane
            X_openings = proj_pt2plane(plane.params, X_openings)
            #Make it homogeneus
            X_op_hom = np.concatenate((X_openings, np.ones((len(X_openings), 1))), axis=1).T
            #Transform it to local coordinates (Z=0)
            Xl, T = open2local(X_op_hom, np.array(self.LOD2.mesh.triangle_normals), plane.params[:3]) 
            
            #Create opening representation as rectangle [x1,y1,x2,y2] (x1,y1)bl, (x2,y2)tr
            x_openings_rectangles = []
            x_openings_rectangles_diag = []
            plt.figure()
            for ii in range(len(plane.openings)):
                X_op_rect = (Xl.T)[4*ii:4*ii+4]
                x_openings_rectangles.append([np.min(X_op_rect[:,0]), np.min(X_op_rect[:,1]), np.max(X_op_rect[:,0]), np.max(X_op_rect[:,1])])
                x_openings_rectangles_diag.append((np.min(X_op_rect[:,0])-np.max(X_op_rect[:,0]))**2 + (np.min(X_op_rect[:,1])-np.max(X_op_rect[:,1]))**2)
                plt.scatter(X_op_rect[:,0], X_op_rect[:,1])
            #plt.show()
    
            x_openings_rectangles = np.array(x_openings_rectangles)
            x_openings_rectangles_diag = np.array(x_openings_rectangles_diag)
            #Loop to check if the rectangles overlap
            id_redundant = []
            for ii in range(len(plane.openings)):
                if ii in id_redundant:
                    continue
                id_overlap = []
                for jj in range(ii+1,len(plane.openings)):
                    if rect_overlap(x_openings_rectangles[ii], x_openings_rectangles[jj]):
                        id_overlap.append(jj)
                if len(id_overlap)>0:
                    id_overlap.append(ii)
                    id_overlap = np.array(id_overlap)
                    if parameter=="size":
                        #Select the one with bigger dimentionss
                        diag_rect = x_openings_rectangles_diag[id_overlap]
                        id_redundant += id_overlap[np.where(diag_rect!=np.max(diag_rect))[0]].tolist()
                    elif parameter=="camera":
                        #Select the closest to the camera
                        d2int_op = np.mean(np.array([plane.openings[kk].d2int for kk in id_overlap]), axis=1)
                        id_redundant += id_overlap[np.where(d2int_op!=np.min(d2int_op))[0]].tolist()
                    elif parameter=="camera-size":
                        #If the difference between sizes is too big, select the one with biggest size
                        diag_rect = x_openings_rectangles_diag[id_overlap]
                        relative_size = np.abs((diag_rect - diag_rect[0])/diag_rect[0])
                        d2int_op = np.mean(np.array([plane.openings[kk].d2int for kk in id_overlap]), axis=1)
                        if np.all(relative_size<.4):
                            print("relative size too small, select the biggest opening ", relative_size)
                            id_redundant += id_overlap[np.where(diag_rect!=np.max(diag_rect))[0]].tolist()
                        else:
                            print("redundant openings with different sizes, select the one with closest camera", relative_size)
                            #Select the closest to the camera
                            id_redundant += id_overlap[np.where(d2int_op!=np.min(d2int_op))[0]].tolist()


            
            #Assign redundant label to opening
            for ii in id_redundant:
                plane.openings[ii].redundant=True


                    
    
    def regularize_openings(self, ctes, use_redundant=False):
        #Loop for each plane that contain openings and regularize them using the aligning methods in local coordinates
        for plane in self.LOD2.planes:
            if len(plane.openings)==0:
                continue
            X_openings = np.empty(shape=(0,3))
            for opening in plane.openings:
                if use_redundant:
                    X_openings = np.concatenate((X_openings, opening.coord), axis=0)
                else:
                    if not opening.redundant:
                        X_openings = np.concatenate((X_openings, opening.coord), axis=0)
            #Project all opening's corners to the plane
            X_openings = proj_pt2plane(plane.params, X_openings)
            #Make it homogeneus
            X_op_hom = np.concatenate((X_openings, np.ones((len(X_openings), 1))), axis=1).T
            #Transform it to local coordinates (Z=0)
            Xl, T = open2local(X_op_hom, np.array(self.LOD2.mesh.triangle_normals), plane.params[:3]) 
            #Aligning the width and height of the openings (Aligment 1 --> to linear regression model).
            Xl_al = op_aligning1(Xl, cte = ctes[0])      
            #CLEANING 2.1: aligning  each opening
            #Aligning the width and height of the openings (Aligment 2 --> same width and height) 
            Xl_al2 = op_aligning2(Xl_al, cte = ctes[1])        
            #Equalizing areas
            Xl_al3 = op_aligning3(Xl_al2, cte1 = ctes[2], cte2 = ctes[3])            
            #Taking to global coordinates again
            X_al = np.dot(np.linalg.inv(T),Xl_al3)
            X_al = X_al[:3].T
            count = 0
            for ii, opening in enumerate(plane.openings):
                if use_redundant:
                    opening.coord = X_al[4*ii:4*ii+4]
                else:
                    if not opening.redundant:
                        opening.coord = X_al[4*count:4*count+4]
                        count+=1
                
    
    def save_openings(self, data_folder):
        #Check if directory exists, if not, create it
        check_dir = os.path.isdir('../results/' + data_folder)
        if not check_dir:
            os.makedirs('../results/' + data_folder) 
        cc_vv=1
        for plane in self.LOD2.planes:
            if len(plane.openings)==0:
                continue
            
            #Writing an .obj file with information of the openings for each pics pair
            f = open('../results/' + data_folder + "/openings{}.obj".format(plane.id), "w")
            
            for opening in plane.openings:
                if not opening.redundant:
                    for X in opening.coord:
                        f.write("v {} {} {}\n".format(X[0],X[1],X[2]))
            c_v = 1 #vertices counter. Helper to identify vertices in generated faces
            num_op2create = len(plane.openings) - len([1 for op in plane.openings if op.redundant])
            #for j in range(len(plane.openings)):
            for j in range(num_op2create):
                f.write("f {} {} {}\n".format(c_v,c_v+1,c_v+2))
                f.write("f {} {} {}\n".format(c_v+1,c_v+2,c_v+3))
                c_v += 4
            f.close()
    
            #Writing an .obj file with information of the openings for all of them
            f = open('../results/' + data_folder + "/openings.obj", "a")
            for opening in plane.openings:
                if not opening.redundant:
                    for X in opening.coord:
                        f.write("v {} {} {}\n".format(X[0],X[1],X[2]))
        
            #for j in range(len(plane.openings)):
            for j in range(num_op2create):
                f.write("f {} {} {}\n".format(cc_vv,cc_vv+1,cc_vv+2))
                f.write("f {} {} {}\n".format(cc_vv+1,cc_vv+2,cc_vv+3))
                cc_vv += 4
            f.close()
    
    def save_cracks2d(self, data_folder):
        for crack in self.cracks:
            cracks2d = crack.coord2d
            np.save('../results/'+data_folder+'/cracks2d_{}.npy'.format(crack.view), cracks2d)

    def save_cracks(self, data_folder, kin_n=False, kin_t=False, kin_tn=False): 

        #Creating colormap
        if kin_n or kin_t or kin_tn:
            from matplotlib import cm
            from matplotlib.colors import ListedColormap
            top = cm.get_cmap('Oranges_r', 256)
            bottom = cm.get_cmap('Blues', 256)
            newcolors = np.vstack((top(np.linspace(0, 1, 256)), bottom(np.linspace(0, 1, 256))))
            newcmp = ListedColormap(newcolors, name='OrangeBlue')

        #Check if directory exists, if not, create it
        check_dir = os.path.isdir('../results/' + data_folder)
        if not check_dir:
            os.makedirs('../results/' + data_folder) 
        cc_vv=1
        num_pts=0
        for jj, crack in enumerate(self.cracks):
            #If there is not crack kinematic information, skip
            if kin_n or kin_t or kin_tn:
                if len(crack.kinematics)==0:
                    continue

            #Creating rgb colors for saving kinematics
            if kin_n:
                cr_name = "_kin_n"
                n = crack.kinematics[:,0]
                n[np.where(n>20)] = 20 
                n_cmp_space = 255*(n - 0)/(20-0)
                rgb = [bottom(int(ni)) for ni in n_cmp_space]
            elif kin_t:
                cr_name = "_kin_t"
                t = crack.kinematics[:,1]
                t[np.where(t<-20)] = -20 
                t[np.where(t>20)] = 20 
                t_cmp_space = 511*(t - (-20))/(20-(-20)) 
                rgb = [newcmp(int(ti)) for ti in t_cmp_space]
            elif kin_tn:
                cr_name = "_kin_tn"
                tn = crack.kinematics[:,1]/crack.kinematics[:,0]
                tn[np.where(tn<-2)] = -2 
                tn[np.where(tn>2)] = 2 
                min_tn = np.min(tn)
                max_tn = np.max(tn)
                tn_cmp_space = 511*(tn - (-2) )/(2-(-2)) 
                rgb = [newcmp(int(tni)) for tni in tn_cmp_space]
            else:
                cr_name = ""
                rgb = [(0,0,0) for pt in crack.coord]


            #Writing an .obj file with information of the openings for each pics pair
            f = open('../results/' + data_folder + "/cracks_{}_{}{}.ply".format(jj, crack.view, cr_name), "w") 
            f.write("ply\n\
            format ascii 1.0\n\
            element vertex {}\n\
            property float x\n\
            property float y\n\
            property float z\n\
            property uchar red\n\
            property uchar green\n\
            property uchar blue\n\
            end_header\n".format(crack.coord.shape[0]))

            for ii, pt in enumerate(crack.coord):
                xx = np.around(pt[0],decimals=5)
                yy = np.around(pt[1],decimals=5)
                zz = np.around(pt[2],decimals=5)
                f.write("{} {} {} {} {} {}\n".format(xx,yy,zz,int(255*rgb[ii][0]),int(255*rgb[ii][1]),int(255*rgb[ii][2])))
            f.close()

            num_pts += len(crack.coord)
            f = open('../results/' + data_folder + "/cracks{}.ply".format(cr_name), "a")
            for ii, pt in enumerate(crack.coord):
                xx = np.around(pt[0],decimals=5)
                yy = np.around(pt[1],decimals=5)
                zz = np.around(pt[2],decimals=5)
                f.write("{} {} {} {} {} {}\n".format(xx,yy,zz,int(255*rgb[ii][0]),int(255*rgb[ii][1]),int(255*rgb[ii][2])))
            f.close()

            if jj==len(self.cracks)-1:
                f = open('../results/' + data_folder + "/cracks{}.ply".format(cr_name), "r")
                read_ply = f.readlines()
                read_ply.insert(0,
                "ply\n\
                format ascii 1.0\n\
                element vertex {}\n\
                property float x\n\
                property float y\n\
                property float z\n\
                property uchar red\n\
                property uchar green\n\
                property uchar blue\n\
                end_header\n".format(num_pts))
                f.close()
                
                f = open('../results/' + data_folder + "/cracks{}.ply".format(cr_name), "w")
                f.writelines(read_ply)
                f.close()
                print(num_pts)  


class LOD2:
    def __init__(self, LOD2_mesh) -> None:
        self.mesh = LOD2_mesh
        self.plane_clusters = None
        self.facades = None
        self.roof = None
        self.boundary = None
        self.planes = None
    
    def get_plane_clusters(self):
        LOD2_triangle_clusters, LOD2_cluster_n_triangles, LOD2_cluster_area = cluster_triangles2plane(self)
        LOD2_triangle_clusters =  np.array(LOD2_triangle_clusters)
        LOD2_cluster_n_triangles = np.array(LOD2_cluster_n_triangles)
        LOD2_cluster_area = np.array(LOD2_cluster_area)
        LOD2_cluster_plane_params = []
        LOD2_cluster_plane_meshes = []
        #Creating plane params accordint to cluster - Visualizing clusters
        for c_id in range(len(LOD2_cluster_n_triangles)):
            triangles_no_current_cluster = LOD2_triangle_clusters!=c_id
            LOD2_mesh_current_cluster = copy.deepcopy(self.mesh)
            LOD2_mesh_current_cluster.remove_triangles_by_mask(triangles_no_current_cluster)
            LOD2_mesh_current_cluster.remove_unreferenced_vertices()
            if LOD2_cluster_area[c_id]==0:
                LOD2_cluster_area[c_id] = LOD2_mesh_current_cluster.get_surface_area()
            _, current_plane_params = fit_plane_on_X(np.array(LOD2_mesh_current_cluster.vertices), fit_ransac=False)
            LOD2_cluster_plane_params.append(current_plane_params)
            plane = Plane(LOD2_mesh_current_cluster)
            plane.id = c_id
            plane.params = current_plane_params
            LOD2_cluster_plane_meshes.append(plane)
        LOD2_cluster_plane_params = np.array(LOD2_cluster_plane_params)
        
        self.plane_clusters = [LOD2_triangle_clusters, LOD2_cluster_n_triangles, LOD2_cluster_area, LOD2_cluster_plane_params]
        self.planes = LOD2_cluster_plane_meshes
    
    def get_boundary_line_set(self, plot = True):
        edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=False)
        ls = o3dtut.edges_to_lineset(self.mesh, edges, (0,0,1))
        if plot:
            o3d.visualization.draw_geometries([self.mesh, ls], mesh_show_back_face=True)        
        self.boundary = ls
    
    def get_planes_contour(self):
        for plane in self.planes:
            plane.get_edges()
            plane.get_contour()
        
    
    def get_facades(self):
        pass

    def plot_mesh(self):
        o3d.visualization.draw_geometries([self.mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
 
    def plot_boundary(self):
        o3d.visualization.draw_geometries([self.boundary])
   
    def decimate(self):
        LOD2_mesh_decimated = o3d.geometry.TriangleMesh()
        for plane in self.planes:
            plane.decimate(automatic=True)
            LOD2_mesh_decimated+=plane.mesh
        
        LOD2_mesh_decimated.merge_close_vertices(1e-27)
        self.mesh = LOD2_mesh_decimated
        
        


    def check_properties(self):
        self.mesh.compute_vertex_normals()

        edge_manifold = self.mesh.is_edge_manifold(allow_boundary_edges=True)
        edge_manifold_boundary = self.mesh.is_edge_manifold(allow_boundary_edges=False)
        vertex_manifold = self.mesh.is_vertex_manifold()
        self_intersecting = self.mesh.is_self_intersecting()
        watertight = self.mesh.is_watertight()
        orientable = self.mesh.is_orientable()

        print("LOD2 mesh properties")
        print(f"  edge_manifold:          {edge_manifold}")
        print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
        print(f"  vertex_manifold:        {vertex_manifold}")
        print(f"  self_intersecting:      {self_intersecting}")
        print(f"  watertight:             {watertight}")
        print(f"  orientable:             {orientable}")

        geoms = [self.mesh]
        if not edge_manifold:
            edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=True)
            geoms.append(o3dtut.edges_to_lineset(self.mesh, edges, (1, 0, 0)))
        if not edge_manifold_boundary:
            edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=False)
            geoms.append(o3dtut.edges_to_lineset(self.mesh, edges, (0, 1, 0)))
        if not vertex_manifold:
            verts = np.asarray(self.mesh.get_non_manifold_vertices())
            pcl = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices)[verts]))
            pcl.paint_uniform_color((0, 0, 1))
            geoms.append(pcl)
        if self_intersecting:
            intersecting_triangles = np.asarray(
                self.mesh.get_self_intersecting_triangles())
            intersecting_triangles = intersecting_triangles[0:1]
            intersecting_triangles = np.unique(intersecting_triangles)
            print("  # visualize self-intersecting triangles")
            triangles = np.asarray(self.mesh.triangles)[intersecting_triangles]
            edges = [
                np.vstack((triangles[:, i], triangles[:, j]))
                for i, j in [(0, 1), (1, 2), (2, 0)]
            ]
            edges = np.hstack(edges).T
            edges = o3d.utility.Vector2iVector(edges)
            geoms.append(o3dtut.edges_to_lineset(self.mesh, edges, (1, 0, 1)))
        o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

class Opening:

    def __init__(self):
        self.id = -1
        self.facade = -1
        self.plane = -1
        self.coord = None
        self.d2int = [] #!TO CHECK
        self.redundant = False

class Crack:
    def __init__(self):
        self.view = ""
        self.coord2d = None
        self.coord = None
        self.plane = None
        self.id_cr_intersect = None
        self.kinematics = None

class Plane:
    def __init__(self, mesh):
        self.id = -1
        self.mesh = mesh
        self.edges = None
        self.contour = None
        self.openings = []
        self.params = None
    
    def get_edges(self):
        self.edges = trimesh2edges(self.mesh)
    
    def get_contour(self):
        edges = self.mesh.get_non_manifold_edges(allow_boundary_edges=False)
        self.contour = o3dtut.edges_to_lineset(self.mesh, edges, (0,1,0))
    
    def plot_mesh(self):
        o3d.visualization.draw_geometries([self.mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
    
    def plot_contour(self):
        o3d.visualization.draw_geometries([self.contour])
    
    def decimate(self, target_number_of_triangles=5, automatic=False, threshold_area_change = 0.1):
        if not automatic:
            mesh_smp = self.mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
            self.mesh = mesh_smp
        else:
            initial_surface_area = self.mesh.get_surface_area()
            targ_tri = len(self.mesh.triangles)-1
            check_area_change=True
            while check_area_change:
                mesh_smp = self.mesh.simplify_quadric_decimation(target_number_of_triangles=targ_tri)
                targ_tri-=1
                current_surface_area = mesh_smp.get_surface_area()
                area_chage = 100*np.abs((initial_surface_area - current_surface_area)/initial_surface_area)
                if area_chage>threshold_area_change:
                    check_area_change=False
                else:
                    self.mesh = mesh_smp
    
    
        