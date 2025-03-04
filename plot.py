import pyvistaqt as pvqt
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def observation(interface_points, orientation_points, extent, notebook=False):
    """
    visualize the observation points, including interface points and orientation points
    interface_points: interface points of all kinds of structural interfaces, original data from .csv file
    orientation_points: orientation points of all kinds of structural interfaces, original data from .csv file
    extent: the extent of the model
    """
    #p_input = pvqt.BackgroundPlotter()
    p_input = pv.Plotter(notebook=notebook, window_size=(800, 600))
    # color map for different rock types
    # if the number of rock types is more than 9, the color map should be extended
    cmap = ['purple', 'green', 'yellow', 'goldenrod', 'purple', 'green', 'yellow', 'goldenrod', 'orangered']
    # setting for pick the color
    n_layer, n_unconf = 0, 0

    # fault points and orientations
    fault_point = interface_points[interface_points['type'] == 'fault']
    fault_orien = orientation_points[orientation_points['type'] == 'fault']
    p_input.add_points(fault_point[['X', 'Y', 'Z']].values, color='white', point_size=13, render_points_as_spheres=True)
    p_input.add_arrows(fault_orien[['X', 'Y', 'Z']].values, fault_orien[['dx', 'dy', 'dz']].values, mag=(extent[5]-extent[4])/13, color='black')

    # stratigraphic points and orientations
    # stratigraphic points and orientations use different colors
    strat_point = interface_points[interface_points['type'] == 'stratigraphic']
    strat_orien = orientation_points[orientation_points['type'] == 'stratigraphic']
    for n in strat_point['formation'].unique():
        p_input.add_points(strat_point[strat_point['formation'] == n][['X', 'Y', 'Z']].values, 
                           color=cmap[n_layer], point_size=13, render_points_as_spheres=True)
        p_input.add_arrows(strat_orien[strat_orien['formation'] == n][['X', 'Y', 'Z']].values, 
                           strat_orien[strat_orien['formation'] == n][['dx', 'dy', 'dz']].values, color=cmap[n_layer], mag=(extent[5]-extent[4])/13)
        n_layer += 1

    # unconformity points and orientations
    if (interface_points['type'] == 'unconformity').any():
        unconformity_point = interface_points[interface_points['type'] == 'unconformity']
        unconformity_orien = orientation_points[orientation_points['type'] == 'unconformity']
        for m in unconformity_point['formation'].unique():
            n_unconf -= 1
            p_input.add_points(unconformity_point[unconformity_point['formation'] == m][['X', 'Y', 'Z']].values, 
                               color=cmap[n_unconf], point_size=13, render_points_as_spheres=True)
            p_input.add_arrows(unconformity_orien[unconformity_orien['formation'] == m][['X', 'Y', 'Z']].values, 
                               unconformity_orien[unconformity_orien['formation'] == m][['dx', 'dy', 'dz']].values, color=cmap[n_unconf], mag=(extent[5]-extent[4])/13)
            
    # model boundary
    p_input.add_mesh(pv.Box(bounds=[extent[0], extent[1], extent[2], extent[3], extent[4], extent[5]]), color='k', opacity=0.1)
    p_input.add_bounding_box(color='black', line_width=2)
    p_input.add_axes()
    
    return p_input.show()


def observation_fault_mesh(interface_points, orientation_points, extent, fault_mesh_list, notebook=False):
    """
    visualize the observation points and fault meshs
    interface_points: interface points of all kinds of structural interfaces, original data from .csv file
    orientation_points: orientation points of all kinds of structural interfaces, original data from .csv file
    extent: the extent of the model
    """
    #p_input = pvqt.BackgroundPlotter()
    p_input = pv.Plotter(notebook=notebook, window_size=(800, 600))
    # color map for different rock types
    # if the number of rock types is more than 9, the color map should be extended
    cmap = ['purple', 'green', 'yellow', 'goldenrod', 'purple', 'green', 'yellow', 'goldenrod', 'orangered']
    # setting for pick the color
    n_layer, n_unconf = 0, 0

    # fault points and orientations
    fault_point = interface_points[interface_points['type'] == 'fault']
    fault_orien = orientation_points[orientation_points['type'] == 'fault']
    p_input.add_points(fault_point[['X', 'Y', 'Z']].values, color='white', point_size=13, render_points_as_spheres=True)
    p_input.add_arrows(fault_orien[['X', 'Y', 'Z']].values, fault_orien[['dx', 'dy', 'dz']].values, mag=(extent[5]-extent[4])/13, color='black')

    # stratigraphic points and orientations
    # stratigraphic points and orientations use different colors
    strat_point = interface_points[interface_points['type'] == 'stratigraphic']
    strat_orien = orientation_points[orientation_points['type'] == 'stratigraphic']
    for n in strat_point['formation'].unique():
        p_input.add_points(strat_point[strat_point['formation'] == n][['X', 'Y', 'Z']].values, 
                           color=cmap[n_layer], point_size=13, render_points_as_spheres=True)
        p_input.add_arrows(strat_orien[strat_orien['formation'] == n][['X', 'Y', 'Z']].values, 
                           strat_orien[strat_orien['formation'] == n][['dx', 'dy', 'dz']].values, color=cmap[n_layer], mag=(extent[5]-extent[4])/13)
        n_layer += 1

    # unconformity points and orientations
    if (interface_points['type'] == 'unconformity').any():
        unconformity_point = interface_points[interface_points['type'] == 'unconformity']
        unconformity_orien = orientation_points[orientation_points['type'] == 'unconformity']
        for m in unconformity_point['formation'].unique():
            n_unconf -= 1
            p_input.add_points(unconformity_point[unconformity_point['formation'] == m][['X', 'Y', 'Z']].values, 
                               color=cmap[n_unconf], point_size=13, render_points_as_spheres=True)
            p_input.add_arrows(unconformity_orien[unconformity_orien['formation'] == m][['X', 'Y', 'Z']].values, 
                               unconformity_orien[unconformity_orien['formation'] == m][['dx', 'dy', 'dz']].values, color=cmap[n_unconf], mag=(extent[5]-extent[4])/13)

    # fault meshs
    for mesh in fault_mesh_list:
        p_input.add_mesh(mesh, color='lightblue', line_width=5, opacity=0.8)
       
    # model boundary
    p_input.add_bounding_box(color='black', line_width=2.5)
    p_input.add_axes()
    
    return p_input.show()


def fault_mesh(mesh_list, surface_points, orientation_points, extent, notebook=False):
    """
    visualize the fault meshs, all the fault meshs are shown in the same plot
    mesh_list: the list of fault meshs
    surface_points: the observation points, original data from .csv file
    orientation_points: the orientation points, original data from .csv file
    extent: the extent of the model
    """
    #p_fault = pvqt.BackgroundPlotter()
    p_fault = pv.Plotter(notebook=notebook, window_size=(800, 600))
    # color map for different rock types
    # if the number of rock types is more than 9, the color map should be extended
    cmap = ['purple', 'green', 'yellow', 'goldenrod', 'purple', 'green', 'yellow', 'goldenrod', 'orangered']
    # setting for pick the color
    n_layer = 0

    # fault points and orientations
    fault_point = surface_points[surface_points['type'] == 'fault']
    fault_orien = orientation_points[orientation_points['type'] == 'fault']
    p_fault.add_points(fault_point[['X', 'Y', 'Z']].values, color='white', point_size=13, render_points_as_spheres=True)
    p_fault.add_arrows(fault_orien[['X', 'Y', 'Z']].values, fault_orien[['dx', 'dy', 'dz']].values, mag=(extent[5]-extent[4])/13, color='black')

    # stratigraphic points and orientations
    # stratigraphic points and orientations use different colors
    strat_point = surface_points[surface_points['type'] == 'stratigraphic']
    strat_orien = orientation_points[orientation_points['type'] == 'stratigraphic']
    for n in strat_point['formation'].unique():
        p_fault.add_points(strat_point[strat_point['formation'] == n][['X', 'Y', 'Z']].values, 
                           color=cmap[n_layer], point_size=15, render_points_as_spheres=True)
        p_fault.add_arrows(strat_orien[strat_orien['formation'] == n][['X', 'Y', 'Z']].values, 
                           strat_orien[strat_orien['formation'] == n][['dx', 'dy', 'dz']].values, color=cmap[n_layer], mag=(extent[5]-extent[4])/12)
        n_layer += 1
    
    """ # unconformity points and orientations
    p_fault.add_points(surface_points[surface_points['formation'] == 'BCU'][['X', 'Y', 'Z']].values, 
                        color='orangered', point_size=15, render_points_as_spheres=True)
    p_fault.add_arrows(orientation_points[orientation_points['formation'] == 'BCU'][['X', 'Y', 'Z']].values, 
                        orientation_points[orientation_points['formation'] == 'BCU'][['dx', 'dy', 'dz']].values, 
                        color='orangered', mag=(extent[5]-extent[4])/12) """

    # fault meshs
    for mesh in mesh_list:
        p_fault.add_mesh(mesh, color='lightblue', line_width=5, opacity=0.8)
        p_fault.add_bounding_box(color='black', line_width=2)
    
    #p_fault.add_axes()
        
    return p_fault.show()


def feature_encoding(mesh_list, number, extent, type='mesh_point', side=None, domain_mesh_features=None, 
                     label_interf_all=None, orie_points_all=None, notebook=False):
    """
    Notes: 1. visualize the feature encoding results
           2. every time only one fault encoding result is shown
    mesh_list: the list of fault meshs
    number: the index of fault mesh in the mesh_list, number = 0 is the first mesh
    extent: the extent of the model
    type: the type of the encoding, 'mesh_point' or 'obs_point', 'mesh_point' is the meshgrid points in domain for prediction,
            'obs_point' is the observation points
    side: 'up' or 'down', 'up' is the side which assigned value 1 in the feature encoding
    domain_mesh_features: the meshgrid points after feature encoding
    label_interf_all: the observation interface points after feature encoding
    orie_points_all: the orientation points after feature encoding
    """
    # which mesh to visualize
    mesh = mesh_list[number]  # number = 0 is the first mesh
    #p_encoding = pvqt.BackgroundPlotter()
    p_encoding = pv.Plotter(notebook=notebook, window_size=(800, 600))
    p_encoding.add_mesh(mesh, color='grey', line_width=5)
    if type == 'mesh_point':
        if side == 'up':  # relative move up side
            # in feature encoding step, the points on the up side are assigned value 1
            p_encoding.add_points(domain_mesh_features[:, :3][domain_mesh_features[:,3+number]==1], point_size=10, 
                                  render_points_as_spheres=True, color='green')
        else:
            p_encoding.add_points(domain_mesh_features[:, :3][domain_mesh_features[:,3+number]==0], point_size=10, 
                                  render_points_as_spheres=True, color='green')
    else:
        # label_interf_all: [label, X, Y, Z, fault1, fault2...]
        # orie_points_all: [X, Y, Z, dx, dy, dz, fault1, fault2...]
        p_encoding.add_points(label_interf_all[:, 1:4][label_interf_all[:,4+number]==0], point_size=10, render_points_as_spheres=True, color='green')
        p_encoding.add_points(label_interf_all[:, 1:4][label_interf_all[:,4+number]==1], point_size=10, render_points_as_spheres=True, color='red')
        p_encoding.add_arrows(orie_points_all[:, :3][orie_points_all[:,6+number]==0], orie_points_all[:, 3:6][orie_points_all[:,6+number]==0], 
                              mag=(extent[5]-extent[4])/13, color='green')
        p_encoding.add_arrows(orie_points_all[:, :3][orie_points_all[:,6+number]==1], orie_points_all[:, 3:6][orie_points_all[:,6+number]==1],
                              mag=(extent[5]-extent[4])/13, color='red') 
    p_encoding.add_bounding_box(color='black', line_width=1.5)
            
    return p_encoding.show()


def final_structure(fautl_mesh_list, stratigraphic_mesh, unconformity_mesh_list=None, observation_data=False, surface_points=None, 
                    orientation_points=None, extent=None, unconformity=False, notebook=False):
    """
    visualize the final structure, including fault meshs, stratigraphic mesh, unconformity meshs, observations
    fautl_mesh_list: the list of fault meshs
    stratigraphic_mesh: the stratigraphic mesh
    unconformity_mesh_list: the list of unconformity meshs
    observation_data: whether to show the observation data
    surface_points: the observation points, original data from .csv file, if observation_data is True, then surface_points is effective
    orientation_points: the orientation points, original data from .csv file, if observation_data is True, then orientation_points is effective
    extent: the extent of the model
    unconformity: state whether the model has unconformity
    """
    #p_structure = pvqt.BackgroundPlotter()
    p_structure = pv.Plotter(notebook=notebook, window_size=(800, 600))
    # change the color bar 
    scalar_bar_args = {
    'title_font_size': 30,   
    'label_font_size': 30,
    'vertical': False,
    #'n_colors': 256,
    'fmt': '%.2f',
    'position_x': 0.34,
    'position_y': 0.11,
    'width': 0.55,
    'height': 0.09,
    'n_labels': 5,
    'font_family': 'arial',
    }
    # add fault meshs and stratigraphic mesh, if there is unconformity, then add unconformity meshs
    if unconformity:
        for unconform in unconformity_mesh_list:
            p_structure.add_mesh(unconform, color='red', line_width=8, opacity=1)
            new_stratigraphic_mesh = stratigraphic_mesh.clip_surface(unconform, invert=False)
            p_structure.add_mesh(new_stratigraphic_mesh, cmap="viridis", line_width=8, opacity=1, scalar_bar_args=scalar_bar_args)
            for fault in fautl_mesh_list:
                clipped_mesh = fault.clip_box(extent, invert=False)
                new_fault = clipped_mesh.clip_surface(unconform, invert=False)
                p_structure.add_mesh(new_fault, color='white', line_width=5)
    else:
        for fault in fautl_mesh_list:
            clipped_mesh = fault.clip_box(extent, invert=False)
            p_structure.add_mesh(clipped_mesh, color='white', line_width=5)
        p_structure.add_mesh(stratigraphic_mesh, cmap="viridis", line_width=8, opacity=1, scalar_bar_args=scalar_bar_args)

        
    
    # add observation data
    if observation_data:
        # color map for different rock types
        # if the number of rock types is more than 9, the color map should be extended
        cmap = ['purple', 'green', 'yellow', 'goldenrod', 'purple', 'green', 'yellow', 'goldenrod', 'orangered']
        n_layer, n_unconf = 0, 0

        # fault points and orientations
        fault_point = surface_points[surface_points['type'] == 'fault']
        fault_orien = orientation_points[orientation_points['type'] == 'fault']
        p_structure.add_points(fault_point[['X', 'Y', 'Z']].values, color='white', point_size=8, render_points_as_spheres=True)
        p_structure.add_arrows(fault_orien[['X', 'Y', 'Z']].values, fault_orien[['dx', 'dy', 'dz']].values, mag=(extent[5]-extent[4])/13, color='black')

        # stratigraphic points and orientations
        strat_point = surface_points[surface_points['type'] == 'stratigraphic']
        strat_orien = orientation_points[orientation_points['type'] == 'stratigraphic']
        for n in strat_point['formation'].unique():
            p_structure.add_points(strat_point[strat_point['formation'] == n][['X', 'Y', 'Z']].values, 
                                color=cmap[n_layer], point_size=8, render_points_as_spheres=True)
            p_structure.add_arrows(strat_orien[strat_orien['formation'] == n][['X', 'Y', 'Z']].values, 
                                strat_orien[strat_orien['formation'] == n][['dx', 'dy', 'dz']].values, color=cmap[n_layer], mag=(extent[5]-extent[4])/13)
            n_layer += 1

        # unconformity points and orientations
        if (surface_points['type'] == 'unconformity').any():
            unconformity_point = surface_points[surface_points['type'] == 'unconformity']
            unconformity_orien = orientation_points[orientation_points['type'] == 'unconformity']
            for m in unconformity_point['formation'].unique():
                n_unconf -= 1
                p_structure.add_points(unconformity_point[unconformity_point['formation'] == m][['X', 'Y', 'Z']].values, 
                                color=cmap[n_unconf], point_size=8, render_points_as_spheres=True, opacity=1)
                """ p_structure.add_arrows(unconformity_orien[unconformity_orien['formation'] == m][['X', 'Y', 'Z']].values, 
                                unconformity_orien[unconformity_orien['formation'] == m][['dx', 'dy', 'dz']].values, 
                                color=cmap[n_unconf], mag=(extent[5]-extent[4])/13, opacity=0.5) """
            
    p_structure.add_bounding_box(color='black', line_width=3)
    p_structure.add_axes(line_width=3, viewport=(0, 0, 0.25, 0.25))

    return p_structure.show()


def slice(scalar_field, extent, save_image=False):
    """
    visualize the 2D slice of the model, this slice come from the 3D scalar field
    scalar_field: the 3D scalar field
    extent: the extent of the model
    """
    # the location can be parameterized to make more slices
    y_pos = (extent[3]-extent[2])/2  # slice in the middle of y direction
    # slice the scalar field
    slice_field = scalar_field.slice(normal='y', origin=[0, y_pos, 0])

    x = np.linspace(extent[0], extent[1], 100)
    z = np.linspace(extent[4], extent[5], 100)
    X, Z = np.meshgrid(x, z)
    scalar_v = slice_field['scalar'].reshape(X.shape)

    plt.figure(figsize=(6, 3))
    plt.pcolormesh(X, Z, scalar_v, shading='auto', cmap='viridis')
    plt.colorbar(label='Scalar Value')
    plt.xlabel('X_direction')
    plt.ylabel('Z_direction')
    plt.title('Cross section of scalar field at y=500')
    #plt.legend()
    """ plt.xlim(0, 1000)
    plt.ylim(-1000, -400) """
    plt.text(100, -800, '[0, 1]', color='black', fontsize=12)
    plt.text(370, -600, '[0, 0]', color='black', fontsize=12)
    plt.text(700, -800, '[1, 0]', color='black', fontsize=12)
    #plt.grid()

    if save_image:
        plt.savefig('save_images\\slice.png', bbox_inches = 'tight', dpi=300)

    return plt.show()


def layer(scalar_field, fault_mesh_list, extent, notebook=False):
    """
    visualize the single strtigraphic layer, the scalar field is thresholded to get the layer
    scalar_field: the 3D scalar field
    fault_mesh_list: the list of fault
    """
    viridis = plt.cm.get_cmap('viridis', 256)  # full 'viridis' colormap
    # 25% - 75% of the 'viridis' colormap
    middle_colors = viridis(np.linspace(0.3, 0.8, 256))
    custom_cmap = ListedColormap(middle_colors)

    #p_layer = pvqt.BackgroundPlotter()
    p_layer = pv.Plotter(notebook=notebook, window_size=(800, 600))
    subset_mesh = scalar_field.threshold([-1,0], scalars='scalar')

    # change the color bar 
    scalar_bar_args = {
    'title_font_size': 30,   
    'label_font_size': 30,
    'vertical': False,
    #'n_colors': 256,
    'fmt': '%.2f',
    'position_x': 0.34,
    'position_y': 0.11,
    'width': 0.55,
    'height': 0.09,
    'n_labels': 5,
    'font_family': 'arial',
    }

    p_layer.add_mesh(subset_mesh, opacity=1, cmap=custom_cmap, scalar_bar_args=scalar_bar_args)
    for fault in fault_mesh_list:
        clipped_mesh = fault.clip_box(extent, invert=False)
        p_layer.add_mesh(clipped_mesh, color='white', opacity=1)

    p_layer.add_bounding_box(color='black', line_width=3)
    p_layer.add_axes(line_width=3, viewport=(0, 0, 0.25, 0.25))

    """ # get the camera position, focal point and up vector
    camera_position = p_layer.camera.position
    camera_focal_point = p_layer.camera.focal_point
    camera_up_vector = p_layer.camera.up

    print("Camera position:", camera_position)
    print("Camera focal point:", camera_focal_point)
    print("Camera up vector:", camera_up_vector)

    # save the screenshot
    p_layer.screenshot('screenshot.png', window_size=(800, 800), scale=3) """
    
    return p_layer.show()


def scalar_field(scalar_field, fault_mesh, notebook=False):
    """
    visualize the 3D scalar field
    scalar_field: the 3D scalar field
    """
    viridis = plt.cm.get_cmap('viridis', 30)  # Get 'viridis' colormap with 10 bins
    newcolors = viridis(np.linspace(0, 1, 30))  # Generate the colors
    custom_cmap = ListedColormap(newcolors)  # Create a new ListedColormap

    #p_scalar = pvqt.BackgroundPlotter()
    p_scalar = pv.Plotter(notebook=notebook, window_size=(800, 600))
    # change the color bar 
    scalar_bar_args = {
    'title_font_size': 30,   
    'label_font_size': 30,
    'vertical': False,
    #'n_colors': 256,
    'fmt': '%.2f',
    'position_x': 0.34,
    'position_y': 0.11,
    'width': 0.55,
    'height': 0.09,
    'n_labels': 5,
    'font_family': 'arial',
    }
    p_scalar.add_mesh(scalar_field, cmap=custom_cmap, scalar_bar_args=scalar_bar_args)
    
    """ for fault in fault_mesh:
        p_scalar.add_mesh(fault, color='white') """

    p_scalar.add_bounding_box(color='black', line_width=2.0)
    p_scalar.add_axes(line_width=3, viewport=(0, 0, 0.25, 0.25))

    return p_scalar.show()


def stratigraphic(scalar_field, notebook=False):
    """
    visualize the stratigraphic units, the scalar field is thresholded to get the stratigraphic model
    scalar_field: the 3D scalar field
    *** only for case 1 ***
    """
    #p_stratigraphic = pvqt.BackgroundPlotter()
    p_stratigraphic = pv.Plotter(notebook=notebook, window_size=(800, 600))

    scalar = scalar_field.point_data['scalar'] 
    scalar = np.where(scalar >= 1, 4, scalar)
    scalar = np.where((scalar < 1) & (scalar >= 0.33), 3, scalar)
    scalar = np.where((scalar < 0.33) & (scalar >= -0.33), 2, scalar)
    scalar = np.where((scalar < -0.33) & (scalar >= -1), 1, scalar)
    scalar = np.where(scalar < -1, 0, scalar)
    scalar_field.point_data['Rock types'] = scalar.astype(int).ravel()

    viridis = plt.cm.get_cmap('viridis', 256)  # full 'viridis' colormap
    # 25% - 75% of the 'viridis' colormap
    middle_colors = viridis(np.linspace(0.15, 1, 256))
    custom_cmap = ListedColormap(middle_colors)

    # change the color bar 
    scalar_bar_args = {
    'title_font_size': 25,   
    'label_font_size': 25,
    'vertical': False,
    #'n_colors': 256,
    'fmt': '%.2f',
    'position_x': 0.34,
    'position_y': 0.11,
    'width': 0.55,
    'height': 0.08,
    'n_labels': 5,
    'font_family': 'arial',
    }
    p_stratigraphic.add_mesh(scalar_field, opacity=1, scalars='Rock types', cmap=custom_cmap, scalar_bar_args=scalar_bar_args)

    p_stratigraphic.add_bounding_box(color='black', line_width=2.0)
    p_stratigraphic.add_axes(line_width=3, viewport=(0, 0, 0.3, 0.25))

    return p_stratigraphic.show()

